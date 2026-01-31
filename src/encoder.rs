//! Encoder: payload → f32 audio samples (AUDIBLE_FAST protocol).

use std::f64::consts::PI;

use crate::protocol::*;
use crate::reed_solomon::ReedSolomon;

/// Encode a payload into f32 audio samples.
///
/// `volume` is 0..=100 (typically 10-50).
/// Returns a Vec<f32> containing the complete waveform.
pub fn encode(payload: &[u8], volume: u8) -> Result<Vec<f32>, crate::Error> {
    let data_length = payload.len();
    if data_length == 0 {
        return Err(crate::Error::EmptyPayload);
    }
    if data_length > MAX_LENGTH_VARIABLE {
        return Err(crate::Error::PayloadTooLarge {
            size: data_length,
            max: MAX_LENGTH_VARIABLE,
        });
    }
    if volume > 100 {
        return Err(crate::Error::InvalidVolume(volume));
    }

    let send_volume = volume as f64 / 100.0;

    // --- Step 1: Prepare tx_data = [length_byte, payload_bytes...] ---
    // tx_data[0] = length, tx_data[1..=len] = payload
    let mut tx_data = vec![0u8; data_length + 1];
    tx_data[0] = data_length as u8;
    tx_data[1..=data_length].copy_from_slice(payload);

    // --- Step 2: Reed-Solomon encode ---
    // Length RS: encode the length byte with (msg=1, ecc=ENCODED_DATA_OFFSET-1=2)
    let rs_length = ReedSolomon::new(1, ENCODED_DATA_OFFSET - 1);
    let encoded_length = rs_length.encode(&tx_data[0..1]); // [len, ecc0, ecc1]

    // Data RS: encode the payload with appropriate ECC
    let n_ecc = ecc_bytes_for_length(data_length);
    let rs_data = ReedSolomon::new(data_length, n_ecc);
    let encoded_data = rs_data.encode(&tx_data[1..=data_length]);

    // Concatenate: encoded_length (3 bytes) + encoded_data (data_length + n_ecc bytes)
    let mut data_encoded = Vec::with_capacity(ENCODED_DATA_OFFSET + data_length + n_ecc);
    data_encoded.extend_from_slice(&encoded_length);
    data_encoded.extend_from_slice(&encoded_data);

    let total_bytes = data_encoded.len();

    // --- Step 3: Compute total number of data frames ---
    let total_data_frames =
        EXTRA * ((total_bytes + BYTES_PER_TX - 1) / BYTES_PER_TX) * FRAMES_PER_TX;

    let total_frames = N_MARKER_FRAMES + total_data_frames + N_MARKER_FRAMES;

    // --- Step 4: Pre-compute per-tone waveforms ---
    let n_data_bits = 2 * BYTES_PER_TX * 16; // 96 for AUDIBLE_FAST
    let hz_per_sample = HZ_PER_SAMPLE;
    let ihz_per_sample = 1.0 / hz_per_sample;
    let i_samples_per_frame = 1.0 / SAMPLES_PER_FRAME as f64;

    let mut bit1_amplitude = vec![vec![0.0f32; SAMPLES_PER_FRAME]; n_data_bits];
    let mut bit0_amplitude = vec![vec![0.0f32; SAMPLES_PER_FRAME]; n_data_bits];

    for k in 0..n_data_bits {
        let freq = bit_freq(k);
        let freq1 = freq; // skip western-note quantization for simplicity
        let freq0 = freq + hz_per_sample * FREQ_DELTA_BIN as f64;

        let phase_offset = (PI * k as f64) / N_DATA_BITS_PER_TX as f64;
        let beta = 0.35;
        let fmod_cycles = 3.0 + (k % 5) as f64;
        let amp_scale = 0.9 + 0.1 * ((k as f64) * 1.7).sin();

        for i in 0..SAMPLES_PER_FRAME {
            let curi = i as f64;
            let phase1 =
                (2.0 * PI) * (curi * i_samples_per_frame) * (freq1 * ihz_per_sample) + phase_offset;
            let modulator = beta * ((2.0 * PI) * fmod_cycles * (curi * i_samples_per_frame)).sin();
            bit1_amplitude[k][i] = (amp_scale * (phase1 + modulator).sin()) as f32;
        }

        for i in 0..SAMPLES_PER_FRAME {
            let curi = i as f64;
            let phase0 =
                (2.0 * PI) * (curi * i_samples_per_frame) * (freq0 * ihz_per_sample) + phase_offset;
            let modulator = beta * ((2.0 * PI) * fmod_cycles * (curi * i_samples_per_frame)).sin();
            bit0_amplitude[k][i] = (amp_scale * (phase0 + modulator).sin()) as f32;
        }
    }

    // --- Step 5: Generate waveform frame by frame ---
    let mut output = vec![0.0f32; total_frames * SAMPLES_PER_FRAME];

    let frac = 0.05f32;

    let mut frame_id = 0usize;
    while frame_id < total_frames {
        let frame_offset = frame_id * SAMPLES_PER_FRAME;
        let frame_out = &mut output[frame_offset..frame_offset + SAMPLES_PER_FRAME];

        let mut n_freq: usize = 0;

        if frame_id < N_MARKER_FRAMES {
            // --- Start marker ---
            n_freq = N_BITS_IN_MARKER;

            for i in 0..N_BITS_IN_MARKER {
                if i % 2 == 0 {
                    add_amplitude_smooth(
                        &bit1_amplitude[i],
                        frame_out,
                        send_volume as f32,
                        frame_id,
                        N_MARKER_FRAMES,
                        frac,
                    );
                } else {
                    add_amplitude_smooth(
                        &bit0_amplitude[i],
                        frame_out,
                        send_volume as f32,
                        frame_id,
                        N_MARKER_FRAMES,
                        frac,
                    );
                }
            }
        } else if frame_id < N_MARKER_FRAMES + total_data_frames {
            // --- Data frames ---
            let mut data_offset = frame_id - N_MARKER_FRAMES;
            let cycle_mod = data_offset % FRAMES_PER_TX;
            data_offset /= FRAMES_PER_TX;
            data_offset *= BYTES_PER_TX;

            // Determine which bits are active
            let mut data_bits = vec![false; n_data_bits];

            for j in 0..BYTES_PER_TX {
                if data_offset + j < data_encoded.len() {
                    // Low nibble
                    let d_lo = data_encoded[data_offset + j] & 0x0F;
                    data_bits[(2 * j) * 16 + d_lo as usize] = true;

                    // High nibble
                    let d_hi = (data_encoded[data_offset + j] & 0xF0) >> 4;
                    data_bits[(2 * j + 1) * 16 + d_hi as usize] = true;
                }
            }

            for k in 0..n_data_bits {
                if !data_bits[k] {
                    continue;
                }
                n_freq += 1;
                if k % 2 != 0 {
                    add_amplitude_smooth(
                        &bit0_amplitude[k / 2],
                        frame_out,
                        send_volume as f32,
                        cycle_mod,
                        FRAMES_PER_TX,
                        frac,
                    );
                } else {
                    add_amplitude_smooth(
                        &bit1_amplitude[k / 2],
                        frame_out,
                        send_volume as f32,
                        cycle_mod,
                        FRAMES_PER_TX,
                        frac,
                    );
                }
            }
        } else {
            // --- End marker ---
            n_freq = N_BITS_IN_MARKER;
            let f_id = frame_id - (N_MARKER_FRAMES + total_data_frames);

            for i in 0..N_BITS_IN_MARKER {
                if i % 2 == 0 {
                    add_amplitude_smooth(
                        &bit0_amplitude[i],
                        frame_out,
                        send_volume as f32,
                        f_id,
                        N_MARKER_FRAMES,
                        frac,
                    );
                } else {
                    add_amplitude_smooth(
                        &bit1_amplitude[i],
                        frame_out,
                        send_volume as f32,
                        f_id,
                        N_MARKER_FRAMES,
                        frac,
                    );
                }
            }
        }

        // Normalize
        if n_freq == 0 {
            n_freq = 1;
        }
        let scale = 1.0 / n_freq as f32;
        for s in frame_out.iter_mut() {
            *s *= scale;
        }

        frame_id += 1;
    }

    Ok(output)
}

/// Add source amplitude to destination with smooth fade in/out envelope.
///
/// Matches the C++ `addAmplitudeSmooth` function.
fn add_amplitude_smooth(
    src: &[f32],
    dst: &mut [f32],
    scalar: f32,
    cycle_mod: usize,
    n_per_cycle: usize,
    frac: f32,
) {
    let final_id = src.len();
    let n_total = (n_per_cycle * final_id) as f32;
    let ds = frac * n_total;
    let ids = 1.0 / ds;
    let n_begin = (frac * n_total) as usize;
    let n_end = ((1.0 - frac) * n_total) as usize;

    for i in 0..final_id {
        let k = cycle_mod * final_id + i;
        let envelope = if k < n_begin {
            (k as f32) * ids
        } else if k > n_end {
            (n_total - k as f32) * ids
        } else {
            1.0
        };
        dst[i] += scalar * src[i] * envelope;
    }
}
