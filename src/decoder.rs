//! Decoder: f32 audio samples → payload (AUDIBLE_FAST protocol, streaming).

use crate::fft;
use crate::protocol::*;
use crate::reed_solomon::ReedSolomon;

/// Streaming decoder for AUDIBLE_FAST.
pub struct Decoder {
    /// Rolling amplitude history for spectrum averaging.
    amplitude_history: Vec<Vec<f32>>,
    history_id: usize,

    /// Current state.
    state: DecoderState,

    /// Recorded amplitude data during receiving.
    amplitude_recorded: Vec<f32>,
    frames_to_record: usize,
    frames_left_to_record: usize,
    recv_duration_frames: usize,

    /// Marker detection state.
    n_markers_success: usize,
    marker_freq_start: usize,

    /// Accumulation buffer for partial frames.
    sample_buffer: Vec<f32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DecoderState {
    Listening,
    Receiving,
    Analyzing,
}

impl Default for Decoder {
    fn default() -> Self {
        Self::new()
    }
}

impl Decoder {
    pub fn new() -> Self {
        Self {
            amplitude_history: vec![vec![0.0f32; SAMPLES_PER_FRAME]; MAX_SPECTRUM_HISTORY],
            history_id: 0,
            state: DecoderState::Listening,
            amplitude_recorded: Vec::new(),
            frames_to_record: 0,
            frames_left_to_record: 0,
            recv_duration_frames: 0,
            n_markers_success: 0,
            marker_freq_start: 0,
            sample_buffer: Vec::new(),
        }
    }

    /// Reset the decoder to initial state.
    pub fn reset(&mut self) {
        *self = Self::new();
    }

    /// Feed audio samples and attempt to decode.
    ///
    /// Returns `Ok(Some(payload))` when a complete message is decoded,
    /// `Ok(None)` when more data is needed, or `Err` on decode failure.
    pub fn decode(&mut self, samples: &[f32]) -> Result<Option<Vec<u8>>, crate::Error> {
        // Append incoming samples to buffer
        self.sample_buffer.extend_from_slice(samples);

        // Process full frames
        while self.sample_buffer.len() >= SAMPLES_PER_FRAME {
            let frame: Vec<f32> = self.sample_buffer[..SAMPLES_PER_FRAME].to_vec();
            self.sample_buffer.drain(..SAMPLES_PER_FRAME);

            let result = self.process_frame(&frame)?;
            if result.is_some() {
                return Ok(result);
            }
        }

        Ok(None)
    }

    /// Process a single frame of SAMPLES_PER_FRAME samples.
    fn process_frame(&mut self, amplitude: &[f32]) -> Result<Option<Vec<u8>>, crate::Error> {
        // Store in history
        self.amplitude_history[self.history_id][..SAMPLES_PER_FRAME]
            .copy_from_slice(&amplitude[..SAMPLES_PER_FRAME]);
        self.history_id = (self.history_id + 1) % MAX_SPECTRUM_HISTORY;

        // Compute averaged spectrum (when history wraps or receiving)
        let compute_spectrum =
            self.history_id == 0 || self.state == DecoderState::Receiving;

        let spectrum = if compute_spectrum {
            // Average amplitude history
            let mut avg = vec![0.0f32; SAMPLES_PER_FRAME];
            for hist in &self.amplitude_history {
                for (i, &v) in hist.iter().enumerate() {
                    avg[i] += v;
                }
            }
            let norm = 1.0 / MAX_SPECTRUM_HISTORY as f32;
            for v in avg.iter_mut() {
                *v *= norm;
            }

            // FFT -> power spectrum
            let mut spec = vec![0.0f32; SAMPLES_PER_FRAME];
            fft::power_spectrum(&avg, &mut spec);
            Some(spec)
        } else {
            None
        };

        // Record frames during receiving
        if self.state == DecoderState::Receiving && self.frames_left_to_record > 0 {
            let offset =
                (self.frames_to_record - self.frames_left_to_record) * SAMPLES_PER_FRAME;
            if offset + SAMPLES_PER_FRAME <= self.amplitude_recorded.len() {
                self.amplitude_recorded[offset..offset + SAMPLES_PER_FRAME]
                    .copy_from_slice(amplitude);
            }
            self.frames_left_to_record -= 1;
            if self.frames_left_to_record == 0 {
                self.state = DecoderState::Analyzing;
            }
        }

        // Analyze recorded data
        if self.state == DecoderState::Analyzing {
            let result = self.analyze();
            self.state = DecoderState::Listening;
            self.frames_to_record = 0;
            self.n_markers_success = 0;
            return result;
        }

        // Marker detection
        if let Some(ref spec) = spectrum {
            if self.state == DecoderState::Listening {
                self.detect_start_marker(spec);
            } else if self.state == DecoderState::Receiving {
                self.detect_end_marker(spec);
            }
        }

        Ok(None)
    }

    /// Check for start marker pattern in spectrum.
    fn detect_start_marker(&mut self, spectrum: &[f32]) {
        let mut n_detected = N_BITS_IN_MARKER;

        for i in 0..N_BITS_IN_MARKER {
            let freq = bit_freq(i);
            let bin = (freq * IHZ_PER_SAMPLE).round() as usize;

            if bin + FREQ_DELTA_BIN >= spectrum.len() {
                n_detected = 0;
                break;
            }

            // Start marker: even bits have bit1 (higher power at bin),
            // odd bits have bit0 (higher power at bin + delta)
            if i % 2 == 0 {
                if spectrum[bin]
                    <= SOUND_MARKER_THRESHOLD as f32 * spectrum[bin + FREQ_DELTA_BIN]
                {
                    n_detected -= 1;
                }
            } else if spectrum[bin]
                >= SOUND_MARKER_THRESHOLD as f32 * spectrum[bin + FREQ_DELTA_BIN]
            {
                n_detected -= 1;
            }
        }

        if n_detected == N_BITS_IN_MARKER {
            self.marker_freq_start = FREQ_START;
            self.n_markers_success += 1;

            if self.n_markers_success >= 1 {
                self.start_receiving();
            }
        } else {
            self.n_markers_success = 0;
        }
    }

    /// Check for end marker pattern in spectrum.
    fn detect_end_marker(&mut self, spectrum: &[f32]) {
        let mut n_detected = N_BITS_IN_MARKER;

        for i in 0..N_BITS_IN_MARKER {
            let freq = bit_freq(i);
            let bin = (freq * IHZ_PER_SAMPLE).round() as usize;

            if bin + FREQ_DELTA_BIN >= spectrum.len() {
                n_detected = 0;
                break;
            }

            // End marker: inverted pattern from start marker
            if i % 2 == 0 {
                if spectrum[bin]
                    >= SOUND_MARKER_THRESHOLD as f32 * spectrum[bin + FREQ_DELTA_BIN]
                {
                    n_detected -= 1;
                }
            } else if spectrum[bin]
                <= SOUND_MARKER_THRESHOLD as f32 * spectrum[bin + FREQ_DELTA_BIN]
            {
                n_detected -= 1;
            }
        }

        if n_detected == N_BITS_IN_MARKER {
            self.n_markers_success += 1;

            if self.n_markers_success >= 1 && self.frames_to_record > 1 {
                self.recv_duration_frames -= self.frames_left_to_record.saturating_sub(1);
                self.frames_left_to_record = 1;
                self.n_markers_success = 0;
            }
        } else {
            self.n_markers_success = 0;
        }
    }

    /// Transition to receiving state.
    fn start_receiving(&mut self) {
        self.state = DecoderState::Receiving;
        self.n_markers_success = 0;

        // Max receive duration
        let max_total_bytes =
            MAX_LENGTH_VARIABLE + ecc_bytes_for_length(MAX_LENGTH_VARIABLE);
        let max_data_frames =
            FRAMES_PER_TX * ((max_total_bytes / BYTES_PER_TX) + 1);
        self.recv_duration_frames = 2 * N_MARKER_FRAMES + max_data_frames;

        self.frames_to_record = self.recv_duration_frames;
        self.frames_left_to_record = self.recv_duration_frames;
        self.amplitude_recorded =
            vec![0.0f32; self.recv_duration_frames * SAMPLES_PER_FRAME];
    }

    /// Analyze recorded data to extract payload.
    fn analyze(&mut self) -> Result<Option<Vec<u8>>, crate::Error> {
        let steps_per_frame: usize = 16;
        let step = SAMPLES_PER_FRAME / steps_per_frame;

        // Only try AUDIBLE_FAST (freqStart == marker_freq_start)
        if self.marker_freq_start != FREQ_START {
            return Ok(None);
        }

        let mut spectrum = vec![0.0f32; SAMPLES_PER_FRAME];

        // Try each sub-frame offset
        for ii in (0..N_MARKER_FRAMES * steps_per_frame).rev() {
            let offset_start = ii;

            let mut data_encoded = vec![0u8; MAX_DATA_SIZE];
            let mut known_length = false;
            let mut decoded_length: usize = 0;

            for itx in 0..1024 {
                let offset_tx =
                    offset_start + itx * FRAMES_PER_TX * steps_per_frame;

                if offset_tx >= self.recv_duration_frames * steps_per_frame {
                    break;
                }
                if (itx + 1) * BYTES_PER_TX >= MAX_DATA_SIZE {
                    break;
                }

                // Sum frames for this TX chunk
                let sample_start = offset_tx * step;
                if sample_start + SAMPLES_PER_FRAME > self.amplitude_recorded.len() {
                    break;
                }

                let mut fft_buf = vec![0.0f32; SAMPLES_PER_FRAME];
                fft_buf.copy_from_slice(
                    &self.amplitude_recorded[sample_start..sample_start + SAMPLES_PER_FRAME],
                );

                for k in 1..FRAMES_PER_TX {
                    let koffset = (offset_tx + k * steps_per_frame) * step;
                    if koffset + SAMPLES_PER_FRAME > self.amplitude_recorded.len() {
                        break;
                    }
                    for i in 0..SAMPLES_PER_FRAME {
                        fft_buf[i] += self.amplitude_recorded[koffset + i];
                    }
                }

                // FFT -> power spectrum
                fft::power_spectrum(&fft_buf, &mut spectrum);

                // Extract nibbles
                for i in 0..(2 * BYTES_PER_TX) {
                    let freq = HZ_PER_SAMPLE * FREQ_START as f64;
                    let bin = (freq * IHZ_PER_SAMPLE).round() as usize + 16 * i;

                    let mut kmax = 0usize;
                    let mut amax = 0.0f32;
                    for k in 0..16 {
                        if bin + k < spectrum.len() && spectrum[bin + k] > amax {
                            kmax = k;
                            amax = spectrum[bin + k];
                        }
                    }

                    if i % 2 != 0 {
                        let byte_idx = itx * BYTES_PER_TX + i / 2;
                        if byte_idx < data_encoded.len() {
                            data_encoded[byte_idx] |= (kmax as u8) << 4;
                        }
                    } else {
                        let byte_idx = itx * BYTES_PER_TX + i / 2;
                        if byte_idx < data_encoded.len() {
                            // Clear and set low nibble
                            data_encoded[byte_idx] =
                                (data_encoded[byte_idx] & 0xF0) | (kmax as u8);
                        }
                    }
                }

                // Try to decode length after we have enough data
                if itx * BYTES_PER_TX > ENCODED_DATA_OFFSET && !known_length {
                    let rs_length = ReedSolomon::new(1, ENCODED_DATA_OFFSET - 1);
                    if let Some(decoded) = rs_length.decode(&data_encoded[..ENCODED_DATA_OFFSET]) {
                        let len = decoded[0] as usize;
                        if len > 0 && len <= MAX_LENGTH_VARIABLE {
                            // Validate frame count
                            let n_total_bytes_expected =
                                ENCODED_DATA_OFFSET + len + ecc_bytes_for_length(len);
                            let n_total_frames_expected = 2 * N_MARKER_FRAMES
                                + ((n_total_bytes_expected + BYTES_PER_TX - 1) / BYTES_PER_TX)
                                    * FRAMES_PER_TX;

                            if self.recv_duration_frames <= n_total_frames_expected
                                && self.recv_duration_frames
                                    >= n_total_frames_expected - 2 * N_MARKER_FRAMES
                            {
                                known_length = true;
                                decoded_length = len;
                            } else {
                                break;
                            }
                        } else {
                            break;
                        }
                    } else {
                        break;
                    }
                }

                // Stop reading if we have enough bytes
                if known_length {
                    let n_total_bytes_expected =
                        ENCODED_DATA_OFFSET + decoded_length + ecc_bytes_for_length(decoded_length);
                    if itx * BYTES_PER_TX > n_total_bytes_expected + 1 {
                        break;
                    }
                }
            }

            if known_length && decoded_length > 0 {
                let n_ecc = ecc_bytes_for_length(decoded_length);
                let rs_data = ReedSolomon::new(decoded_length, n_ecc);

                let data_start = ENCODED_DATA_OFFSET;
                let data_end = data_start + decoded_length + n_ecc;

                if data_end <= data_encoded.len() {
                    if let Some(decoded) = rs_data.decode(&data_encoded[data_start..data_end]) {
                        return Ok(Some(decoded));
                    }
                }
            }
        }

        Err(crate::Error::DecodeFailed)
    }
}
