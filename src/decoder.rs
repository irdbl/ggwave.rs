//! Decoder: f32 audio samples → payload (2-channel orthogonal vocal modem, streaming).

use crate::fft;
use crate::formant;
use crate::protocol::*;
use crate::reed_solomon::ReedSolomon;

/// Preamble vowel patterns for start detection (vowel-only, robust to misalignment).
const PREAMBLE_START_VOWELS: [usize; PREAMBLE_LEN] = [0, 3, 0, 3];

/// Preamble vowel patterns for end detection.
const PREAMBLE_END_VOWELS: [usize; PREAMBLE_LEN] = [3, 0, 3, 0];

/// End preamble full symbol values for precise matching in analyze().
const PREAMBLE_END_SYMBOLS: [usize; PREAMBLE_LEN] = [6, 0, 6, 0];

/// Number of history windows to keep during Listening for alignment recovery.
const HISTORY_WINDOWS: usize = 6;

/// Streaming decoder for the 2-channel orthogonal vocal modem.
pub struct Decoder {
    state: DecoderState,
    sample_buffer: Vec<f32>,
    recent_vowels: Vec<usize>,
    recent_symbols: Vec<usize>,
    recorded_audio: Vec<f32>,
    symbols_recorded: usize,
    max_record_symbols: usize,
    /// Ring buffer of recent windows during Listening for alignment recovery.
    history: Vec<Vec<f32>>,
    /// Accumulated classified symbols during Receiving (for best-effort preview).
    preview_symbols: Vec<usize>,
    /// Counter for consecutive silence windows (for gap detection).
    silence_count: usize,
    /// Counter for consecutive active windows without preamble match.
    active_count: usize,
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
            state: DecoderState::Listening,
            sample_buffer: Vec::new(),
            recent_vowels: Vec::new(),
            recent_symbols: Vec::new(),
            recorded_audio: Vec::new(),
            symbols_recorded: 0,
            max_record_symbols: 0,
            history: Vec::new(),
            preview_symbols: Vec::new(),
            silence_count: 0,
            active_count: 0,
        }
    }

    pub fn reset(&mut self) {
        *self = Self::new();
    }

    /// Best-effort preview of symbols accumulated during the current Receiving phase.
    ///
    /// Converts classified (but uncorrected) symbols to raw bytes. Returns None
    /// if the decoder is not currently receiving or has too few symbols.
    pub fn preview_bytes(&self) -> Option<Vec<u8>> {
        if self.state != DecoderState::Receiving || self.preview_symbols.is_empty() {
            return None;
        }
        let max_bytes = (self.preview_symbols.len() * BITS_PER_SYMBOL) / 8;
        if max_bytes == 0 {
            return None;
        }
        Some(formant::symbols_to_bytes(&self.preview_symbols, max_bytes))
    }

    pub fn decode(&mut self, samples: &[f32]) -> Result<Option<Vec<u8>>, crate::Error> {
        self.sample_buffer.extend_from_slice(samples);

        while self.sample_buffer.len() >= SYMBOL_TOTAL_SAMPLES {
            let window: Vec<f32> = self.sample_buffer[..SYMBOL_TOTAL_SAMPLES].to_vec();
            self.sample_buffer.drain(..SYMBOL_TOTAL_SAMPLES);

            let result = self.process_symbol_window(&window)?;
            if result.is_some() {
                return Ok(result);
            }
        }

        Ok(None)
    }

    fn process_symbol_window(
        &mut self,
        window: &[f32],
    ) -> Result<Option<Vec<u8>>, crate::Error> {
        match self.state {
            DecoderState::Listening => {
                let vowel = self.classify_window_vowel(window);

                // Track silence for gap detection. When we see sustained silence,
                // clear state so the next audio frame gets clean preamble detection.
                if vowel == NUM_VOWELS {
                    self.silence_count += 1;
                    self.active_count = 0;
                    // After ~4 windows of silence (~60ms at any sample rate),
                    // clear history and recent_vowels to reset alignment.
                    if self.silence_count >= 4 {
                        self.history.clear();
                        self.recent_vowels.clear();
                    }
                } else {
                    self.silence_count = 0;
                    self.active_count += 1;
                }

                // Keep history for alignment recovery
                self.history.push(window.to_vec());
                if self.history.len() > HISTORY_WINDOWS {
                    self.history.remove(0);
                }

                // Only track non-silence vowels for preamble detection.
                // This makes detection robust to misaligned windows at frame boundaries.
                if vowel != NUM_VOWELS {
                    self.recent_vowels.push(vowel);
                    if self.recent_vowels.len() > PREAMBLE_LEN {
                        self.recent_vowels
                            .drain(..self.recent_vowels.len() - PREAMBLE_LEN);
                    }

                    if self.recent_vowels.len() == PREAMBLE_LEN
                        && self.recent_vowels[..] == PREAMBLE_START_VOWELS[..]
                    {
                        self.start_receiving();
                    }
                }

                // Fallback: if we've seen many active windows without preamble match,
                // force into receiving mode. The multi-offset analysis will find alignment.
                // This handles misaligned frame boundaries (e.g., after dropped frame).
                // Threshold must be higher than preamble length (4 windows) to avoid false triggers.
                // 12 windows ≈ 720ms is safely above typical message lengths.
                if self.active_count >= 12 && self.history.len() >= HISTORY_WINDOWS {
                    self.start_receiving();
                }

                Ok(None)
            }
            DecoderState::Receiving => {
                self.recorded_audio.extend_from_slice(window);
                self.symbols_recorded += 1;

                let voiced = &window[..SAMPLES_PER_SYMBOL.min(window.len())];
                let (sym, _) = self.classify_voiced_segment(voiced);
                self.preview_symbols.push(sym);
                let vowel = formant::symbol_vowel(sym);
                self.recent_vowels.push(vowel);
                if self.recent_vowels.len() > PREAMBLE_LEN {
                    self.recent_vowels
                        .drain(..self.recent_vowels.len() - PREAMBLE_LEN);
                }
                self.recent_symbols.push(sym);
                if self.recent_symbols.len() > PREAMBLE_LEN {
                    self.recent_symbols
                        .drain(..self.recent_symbols.len() - PREAMBLE_LEN);
                }

                let min_data_symbols = formant::symbols_for_bytes(ENCODED_DATA_OFFSET);
                // Check full symbol match for end preamble (stronger than vowel-only)
                let end_detected = (self.recent_symbols.len() == PREAMBLE_LEN
                    && self.recent_symbols[..] == PREAMBLE_END_SYMBOLS[..])
                    || (self.recent_vowels.len() == PREAMBLE_LEN
                        && self.recent_vowels[..] == PREAMBLE_END_VOWELS[..]);
                if end_detected
                    && self.symbols_recorded > min_data_symbols + PREAMBLE_LEN
                {
                    let trial_n = self.symbols_recorded - PREAMBLE_LEN;
                    if trial_n >= min_data_symbols {
                        if let Some(payload) = self.trial_decode(trial_n) {
                            self.state = DecoderState::Listening;
                            self.recent_vowels.clear();
                            self.recent_symbols.clear();
                            return Ok(Some(payload));
                        }
                        // Trial decode failed — likely misaligned due to history
                        // prepend, or a false positive vowel/symbol match from data.
                        // Don't transition to Analyzing on first false positive;
                        // continue receiving to find the real end preamble.
                    }
                }

                if self.symbols_recorded >= self.max_record_symbols {
                    self.state = DecoderState::Analyzing;
                    return self.analyze();
                }

                Ok(None)
            }
            DecoderState::Analyzing => Ok(None),
        }
    }

    fn classify_window_vowel(&self, window: &[f32]) -> usize {
        let voiced = &window[..SAMPLES_PER_SYMBOL.min(window.len())];

        let rms = {
            let sum_sq: f64 = voiced.iter().map(|&s| (s as f64) * (s as f64)).sum();
            (sum_sq / voiced.len() as f64).sqrt()
        };
        if rms < 1e-6 {
            return NUM_VOWELS;
        }

        let mut spectrum = vec![0.0f32; voiced.len()];
        fft::power_spectrum_raw(voiced, &mut spectrum);
        let f0 = formant::detect_f0(&spectrum);
        let f2 = formant::detect_formants(&spectrum, f0);
        let (vowel_idx, _) = formant::classify_vowel_f2(f2);
        vowel_idx
    }

    fn classify_voiced_segment(&self, voiced: &[f32]) -> (usize, f64) {
        let mut spectrum = vec![0.0f32; voiced.len()];
        fft::power_spectrum(voiced, &mut spectrum);
        formant::classify_symbol(&spectrum)
    }

    fn start_receiving(&mut self) {
        self.state = DecoderState::Receiving;
        self.recent_vowels.clear();
        self.recent_symbols.clear();
        self.preview_symbols.clear();

        // Prepend history windows to recording for alignment recovery
        self.recorded_audio.clear();
        let n_history = self.history.len();
        for w in &self.history {
            self.recorded_audio.extend_from_slice(w);
        }
        self.symbols_recorded = n_history;
        self.history.clear();

        let max_total_bytes =
            ENCODED_DATA_OFFSET + MAX_LENGTH_VARIABLE + ecc_bytes_for_length(MAX_LENGTH_VARIABLE);
        let max_data_syms = formant::symbols_for_bytes(max_total_bytes);
        self.max_record_symbols = max_data_syms + PREAMBLE_LEN + n_history + 10;

        self.recorded_audio
            .reserve(self.max_record_symbols * SYMBOL_TOTAL_SAMPLES);
    }

    fn trial_decode(&self, n_data_symbols: usize) -> Option<Vec<u8>> {
        let mut symbols = Vec::with_capacity(n_data_symbols);
        let mut confidences = Vec::with_capacity(n_data_symbols);

        for i in 0..n_data_symbols {
            let s = i * SYMBOL_TOTAL_SAMPLES;
            if s + SAMPLES_PER_SYMBOL > self.recorded_audio.len() {
                return None;
            }
            let voiced = &self.recorded_audio[s..s + SAMPLES_PER_SYMBOL];
            let (sym, conf) = self.classify_voiced_segment(voiced);
            symbols.push(sym);
            confidences.push(conf);
        }

        // Skip past start preamble if present in the recorded symbols
        // (history windows prepended in start_receiving may include it)
        let min_data_symbols = formant::symbols_for_bytes(ENCODED_DATA_OFFSET);
        let mut data_start = 0;
        if symbols.len() >= PREAMBLE_LEN + min_data_symbols {
            for pos in 0..symbols.len().saturating_sub(min_data_symbols + PREAMBLE_LEN) {
                let vowels: Vec<usize> = symbols[pos..pos + PREAMBLE_LEN]
                    .iter()
                    .map(|&s| formant::symbol_vowel(s))
                    .collect();
                if vowels == PREAMBLE_START_VOWELS {
                    data_start = pos + PREAMBLE_LEN;
                    break;
                }
            }
        }

        self.try_decode_symbols(&symbols[data_start..], Some(&confidences[data_start..]))
    }

    fn analyze(&mut self) -> Result<Option<Vec<u8>>, crate::Error> {
        let min_data_symbols = formant::symbols_for_bytes(ENCODED_DATA_OFFSET);
        if self.symbols_recorded < min_data_symbols {
            self.state = DecoderState::Listening;
            self.recent_vowels.clear();
            return Err(crate::Error::DecodeFailed);
        }

        let total_audio_len = self.recorded_audio.len();
        let steps_per_symbol: usize = 16;
        let step = SYMBOL_TOTAL_SAMPLES / steps_per_symbol;

        let mut best_payload: Option<Vec<u8>> = None;
        let mut best_confidence: f64 = 0.0;

        // Try more offsets to handle misalignment from padding
        let max_offset_steps = steps_per_symbol * 2;

        for offset_step in 0..max_offset_steps {
            let sample_offset = offset_step * step;

            let mut symbols = Vec::new();
            let mut confidences = Vec::new();
            let mut confidence_sum = 0.0f64;
            let mut sym_idx = 0;

            loop {
                let start = sample_offset + sym_idx * SYMBOL_TOTAL_SAMPLES;
                if start + SAMPLES_PER_SYMBOL > total_audio_len {
                    break;
                }

                let voiced = &self.recorded_audio[start..start + SAMPLES_PER_SYMBOL];
                let (sym, conf) = self.classify_voiced_segment(voiced);
                symbols.push(sym);
                confidences.push(conf);
                confidence_sum += conf;
                sym_idx += 1;
            }

            if symbols.len() < min_data_symbols {
                continue;
            }

            // Try to find and strip start preamble from front
            let mut data_start = 0;
            if symbols.len() >= PREAMBLE_LEN {
                for pos in 0..symbols.len().saturating_sub(min_data_symbols + PREAMBLE_LEN) {
                    let vowels: Vec<usize> = symbols[pos..pos + PREAMBLE_LEN]
                        .iter()
                        .map(|&s| formant::symbol_vowel(s))
                        .collect();
                    if vowels == PREAMBLE_START_VOWELS {
                        data_start = pos + PREAMBLE_LEN;
                        break;
                    }
                }
            }

            // Collect candidate end positions for data region:
            // 1. Full 4-bit symbol match (most precise)
            // 2. Vowel-only match (works with degraded envelope/voicing)
            // 3. No end-preamble stripping (fallback)
            let end_preamble_vowels: [usize; PREAMBLE_LEN] = [7, 0, 7, 0];
            let mut end_candidates = Vec::new();

            for pos in (data_start + min_data_symbols..=symbols.len().saturating_sub(PREAMBLE_LEN)).rev() {
                if pos + PREAMBLE_LEN > symbols.len() { continue; }
                if symbols[pos..pos + PREAMBLE_LEN] == PREAMBLE_END_SYMBOLS {
                    end_candidates.push(pos);
                    break;
                }
            }

            for pos in (data_start + min_data_symbols..=symbols.len().saturating_sub(PREAMBLE_LEN)).rev() {
                if pos + PREAMBLE_LEN > symbols.len() { continue; }
                let vowels: Vec<usize> = symbols[pos..pos + PREAMBLE_LEN]
                    .iter()
                    .map(|&s| formant::symbol_vowel(s))
                    .collect();
                if vowels == end_preamble_vowels {
                    if !end_candidates.contains(&pos) {
                        end_candidates.push(pos);
                    }
                    break;
                }
            }

            end_candidates.push(symbols.len());

            for &data_end in &end_candidates {
                let data_symbols = &symbols[data_start..data_end];
                let data_confs = &confidences[data_start..data_end];
                if data_symbols.len() >= min_data_symbols {
                    if let Some(payload) = self.try_decode_symbols(data_symbols, Some(data_confs)) {
                        let conf = confidence_sum / symbols.len().max(1) as f64;
                        if best_payload.is_none() || conf > best_confidence {
                            best_confidence = conf;
                            best_payload = Some(payload);
                        }
                    }
                }

                // Also try without start preamble stripping
                if data_start > 0 {
                    let data_symbols = &symbols[..data_end];
                    let data_confs = &confidences[..data_end];
                    if data_symbols.len() >= min_data_symbols {
                        if let Some(payload) = self.try_decode_symbols(data_symbols, Some(data_confs)) {
                            let conf = confidence_sum / symbols.len().max(1) as f64;
                            if best_payload.is_none() || conf > best_confidence {
                                best_confidence = conf;
                                best_payload = Some(payload);
                            }
                        }
                    }
                }
            }
        }

        self.state = DecoderState::Listening;
        self.recent_vowels.clear();

        if let Some(payload) = best_payload {
            Ok(Some(payload))
        } else {
            Err(crate::Error::DecodeFailed)
        }
    }

    fn try_decode_symbols(
        &self,
        symbols: &[usize],
        confidences: Option<&[f64]>,
    ) -> Option<Vec<u8>> {
        if symbols.is_empty() {
            return None;
        }

        let min_len_symbols = formant::symbols_for_bytes(ENCODED_DATA_OFFSET);
        if symbols.len() < min_len_symbols {
            return None;
        }

        let max_bytes = (symbols.len() * BITS_PER_SYMBOL) / 8;
        let data_encoded = formant::symbols_to_bytes(symbols, max_bytes);

        if data_encoded.len() < ENCODED_DATA_OFFSET {
            return None;
        }

        // Decode length block: try hard-decision first, then erasure fallback
        let rs_length = ReedSolomon::new(1, ENCODED_DATA_OFFSET - 1);
        let decoded_len_byte = rs_length
            .decode(&data_encoded[..ENCODED_DATA_OFFSET])
            .or_else(|| {
                let confs = confidences?;
                let erasures = compute_byte_erasures(confs, 0, ENCODED_DATA_OFFSET, ENCODED_DATA_OFFSET - 1);
                if erasures.is_empty() {
                    return None;
                }
                rs_length.decode_with_erasures(&data_encoded[..ENCODED_DATA_OFFSET], &erasures)
            })?;
        let len = decoded_len_byte[0] as usize;

        if len == 0 || len > MAX_LENGTH_VARIABLE {
            return None;
        }

        let n_ecc = ecc_bytes_for_length(len);
        let expected_total = ENCODED_DATA_OFFSET + len + n_ecc;

        if data_encoded.len() < expected_total {
            return None;
        }

        // Decode payload block: try hard-decision first, then erasure fallback
        let rs_data = ReedSolomon::new(len, n_ecc);
        let payload_block = &data_encoded[ENCODED_DATA_OFFSET..ENCODED_DATA_OFFSET + len + n_ecc];
        let decoded = rs_data.decode(payload_block).or_else(|| {
            let confs = confidences?;
            // Symbol offset for the payload block: ENCODED_DATA_OFFSET bytes = symbols starting
            // at symbol index (ENCODED_DATA_OFFSET * 8 / BITS_PER_SYMBOL)
            let sym_offset = formant::symbols_for_bytes(ENCODED_DATA_OFFSET);
            let erasures = compute_byte_erasures(confs, sym_offset, len + n_ecc, n_ecc);
            if erasures.is_empty() {
                return None;
            }
            rs_data.decode_with_erasures(payload_block, &erasures)
        })?;

        Some(decoded)
    }
}

/// Confidence threshold below which a symbol is considered unreliable.
/// Normalized confidence is 0..1; weaker symbols cluster below ~0.3.
const ERASURE_CONFIDENCE_THRESHOLD: f64 = 0.35;

/// Convert per-symbol confidences to byte-level RS erasure positions.
///
/// Each byte is packed from 2 symbols (high nibble, low nibble).
/// Byte confidence = min of its two symbol confidences.
/// Returns positions of the weakest bytes (below threshold), capped at
/// `ecc_len / 2` to leave room for hard error correction.
fn compute_byte_erasures(
    symbol_confidences: &[f64],
    sym_offset: usize,
    num_bytes: usize,
    ecc_len: usize,
) -> Vec<usize> {
    let max_erasures = ecc_len / 2;
    if max_erasures == 0 {
        return Vec::new();
    }

    // Each byte comes from 2 symbols (4 bits each = 8 bits per byte)
    let mut byte_confs: Vec<(usize, f64)> = (0..num_bytes)
        .map(|byte_idx| {
            let sym_base = sym_offset + byte_idx * 2;
            let c0 = symbol_confidences.get(sym_base).copied().unwrap_or(0.0);
            let c1 = symbol_confidences.get(sym_base + 1).copied().unwrap_or(0.0);
            (byte_idx, c0.min(c1))
        })
        .collect();

    // Sort by confidence ascending (weakest first)
    byte_confs.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    byte_confs
        .iter()
        .filter(|(_, conf)| *conf < ERASURE_CONFIDENCE_THRESHOLD)
        .take(max_erasures)
        .map(|(pos, _)| *pos)
        .collect()
}
