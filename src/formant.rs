//! 2-channel orthogonal vocal modem: F2-only formant + pitch.
//!
//! Optimized for VoIP codecs (Teams, Zoom) that heavily attenuate F1 (400-850 Hz)
//! but preserve F2 (1000-2500 Hz) and boost low frequencies (~200 Hz).
//!
//! Each symbol encodes 4 bits via two independent channels:
//! - Vowel (3 bits): F2 formant frequency only (8 positions, 1000-2500 Hz)
//! - Pitch (1 bit): F0 frequency (208 Hz G#3 / 277 Hz C#4, perfect fourth)

use std::f64::consts::PI;

use crate::protocol::*;

/// Vowel parameters for one symbol.
///
/// Primary detection uses F2-only (VoIP-safe).
/// Hybrid mode also uses F1 as secondary signal when available.
#[derive(Debug, Clone, Copy)]
pub struct VowelParams {
    /// F1 formant frequency (300-850 Hz range) - secondary signal.
    /// Used by hybrid detector when F1 band is available (clean channels).
    pub f1: f64,
    /// F2 formant frequency (1000-2500 Hz range) - primary signal.
    pub f2: f64,
}

/// Pitch class (F0 frequency).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum PitchClass {
    Low = 0,
    High = 1,
}

impl PitchClass {
    pub fn from_index(i: usize) -> Self {
        match i {
            0 => Self::Low,
            1 => Self::High,
            _ => panic!("invalid pitch index: {i}"),
        }
    }

    /// Return the F0 frequency for this pitch class.
    pub fn f0(self) -> f64 {
        match self {
            Self::Low => F0_LOW,
            Self::High => F0_HIGH,
        }
    }
}

/// The 4-vowel alphabet with hybrid F1+F2 detection.
///
/// Primary detection uses F2-only (1000-2500 Hz) which survives VoIP processing.
/// Hybrid mode adds F1 (300-850 Hz) as secondary signal for clean channels.
///
/// F2 positions chosen for distinct harmonic alignment at BOTH F0 values:
/// - F0_LOW=208Hz: H5=1040, H7=1456, H8=1664, H11=2288
/// - F0_HIGH=277Hz: H4=1108, H5=1385, H6=1662, H8=2216
///
/// F1 positions spread across F1 band for maximum separation when available:
/// - Low F1 (~400 Hz): H2@LOW=416, H1.5@HIGH≈415
/// - Mid-low F1 (~520 Hz): H2.5@LOW=520, H2@HIGH=554
/// - Mid-high F1 (~625 Hz): H3@LOW=624, H2.25@HIGH≈623
/// - High F1 (~830 Hz): H4@LOW=832, H3@HIGH=831
pub const VOWELS: [VowelParams; NUM_VOWELS] = [
    VowelParams { f1: 416.0, f2: 1074.0 },  // 0 — low F1, low F2
    VowelParams { f1: 624.0, f2: 1420.0 },  // 1 — mid-high F1, mid-low F2
    VowelParams { f1: 520.0, f2: 1663.0 },  // 2 — mid-low F1, mid-high F2
    VowelParams { f1: 832.0, f2: 2252.0 },  // 3 — high F1, high F2
];

/// Start preamble: vowel0@low, vowel3@low, repeat.
/// Vowels 0 and 3 have maximum F2 separation.
/// Symbol index = vowel*2 + pitch.
pub const PREAMBLE_START: [usize; PREAMBLE_LEN] = [0, 6, 0, 6];

/// End preamble: vowel3@low, vowel0@low, repeat.
pub const PREAMBLE_END: [usize; PREAMBLE_LEN] = [6, 0, 6, 0];

// --- Symbol mapping ---

/// Decompose a 3-bit symbol index into (vowel, pitch) components.
///
/// Bit layout: `[V1 V0 P]`
/// - vowel = bits 2..1 (index >> 1)
/// - pitch = bit 0 (index & 1)
#[inline]
pub fn symbol_to_params(idx: usize) -> (usize, PitchClass) {
    debug_assert!(idx < NUM_SYMBOLS);
    let vowel = (idx >> 1) & 0x3;
    let pitch = idx & 0x1;
    (vowel, PitchClass::from_index(pitch))
}

/// Compose a 4-bit symbol index from (vowel, pitch) components.
#[inline]
pub fn params_to_symbol(vowel: usize, pitch: usize) -> usize {
    debug_assert!(vowel < NUM_VOWELS);
    debug_assert!(pitch < NUM_PITCHES);
    (vowel << 1) | pitch
}

/// Extract vowel index from composite symbol index.
#[inline]
pub fn symbol_vowel(idx: usize) -> usize {
    (idx >> 1) & 0x3
}

// --- Synthesis ---

/// High-frequency rolloff knee (Hz). Harmonics above this are attenuated.
const HF_ROLLOFF_KNEE: f64 = 2400.0;

/// High-frequency rolloff rate (Hz). Lower = steeper rolloff.
const HF_ROLLOFF_RATE: f64 = 600.0;

/// Compute harmonic amplitudes shaped by F2-only Gaussian formant envelope.
///
/// Returns amplitudes for harmonics H1..H16 at the given F0.
/// Uses only F2 band (abandoned F1 for VoIP compatibility).
/// Wider bandwidth (BW2) helps capture energy even with spectral distortion.
/// Applies a gentle high-frequency rolloff above 2400 Hz to prevent clipping.
pub fn harmonic_amplitudes(f1: f64, f2: f64, f0: f64) -> [f64; NUM_HARMONICS] {
    let mut amps = [0.0f64; NUM_HARMONICS];
    for h in 0..NUM_HARMONICS {
        let freq = f0 * (h + 1) as f64;
        // Dual formant shaping: F1 (narrow) + F2 (wide for robustness)
        let g1 = (-(freq - f1).powi(2) / (2.0 * BW1 * BW1)).exp();
        let g2 = (-(freq - f2).powi(2) / (2.0 * BW2 * BW2)).exp();
        // Combine formants: F2 is primary, F1 adds coloring
        let formant_gain = g2.max(g1 * 0.5);

        // Gentle high-frequency rolloff above knee
        let hf_atten = if freq > HF_ROLLOFF_KNEE {
            (-((freq - HF_ROLLOFF_KNEE) / HF_ROLLOFF_RATE)).exp()
        } else {
            1.0
        };

        amps[h] = formant_gain * hf_atten;
    }
    let max = amps.iter().cloned().fold(0.0f64, f64::max);
    if max > 0.0 {
        for a in &mut amps {
            *a /= max;
        }
    }
    amps
}

/// Synthesize one symbol's audio: SAMPLES_PER_SYMBOL of voiced audio.
///
/// Returns exactly SAMPLES_PER_SYMBOL f64 samples (no guard silence).
/// Decomposes symbol index into vowel/pitch, synthesizes harmonics
/// shaped by formants at the pitch-determined F0, applies 5ms fade in/out.
/// Normalizes output to prevent clipping from harmonic summation.
pub fn synthesize_symbol(symbol_idx: usize, volume: f64) -> Vec<f64> {
    let (vowel_idx, pitch_class) = symbol_to_params(symbol_idx);
    let vowel = &VOWELS[vowel_idx];
    let f0 = pitch_class.f0();
    let amps = harmonic_amplitudes(vowel.f1, vowel.f2, f0);

    let fade_samples = (SAMPLE_RATE * 0.005) as usize; // 5ms = 240 samples
    let mut out = vec![0.0f64; SAMPLES_PER_SYMBOL];

    // First pass: synthesize raw waveform
    for i in 0..SAMPLES_PER_SYMBOL {
        let t_abs = i as f64 / SAMPLE_RATE;

        // Harmonic synthesis
        let mut sample = 0.0f64;
        for h in 0..NUM_HARMONICS {
            let freq = f0 * (h + 1) as f64;
            sample += amps[h] * (2.0 * PI * freq * t_abs).sin();
        }

        out[i] = sample;
    }

    // Normalize to prevent clipping (harmonics can sum > 1.0)
    let peak = out.iter().map(|s| s.abs()).fold(0.0f64, f64::max);
    if peak > 0.0 {
        let norm = 1.0 / peak;
        for s in &mut out {
            *s *= norm;
        }
    }

    // Second pass: apply volume and fade envelope
    for i in 0..SAMPLES_PER_SYMBOL {
        let fade = if i < fade_samples {
            i as f64 / fade_samples as f64
        } else if i >= SAMPLES_PER_SYMBOL - fade_samples {
            (SAMPLES_PER_SYMBOL - 1 - i) as f64 / fade_samples as f64
        } else {
            1.0
        };

        out[i] *= volume * fade;
    }

    out
}

// --- Detection ---

/// Detect F0 from a power spectrum by testing candidate F0 values.
///
/// Scores each candidate by summing spectral power at expected harmonic bins.
/// Returns the best-fit F0.
pub fn detect_f0(spectrum: &[f32]) -> f64 {
    let mut best_f0 = F0_LOW;
    let mut best_score = 0.0f64;

    for &f0 in &F0_CANDIDATES {
        let mut score = 0.0f64;
        for h in 1..=NUM_HARMONICS {
            let freq = f0 * h as f64;
            let bin = (freq / HZ_PER_BIN).round() as usize;
            if bin >= spectrum.len() {
                break;
            }
            // Sum power in ±2 bin window
            for offset in 0..=2usize {
                if offset == 0 {
                    score += spectrum[bin] as f64;
                } else {
                    if bin >= offset {
                        score += spectrum[bin - offset] as f64 * 0.5;
                    }
                    if bin + offset < spectrum.len() {
                        score += spectrum[bin + offset] as f64 * 0.5;
                    }
                }
            }
        }
        if score > best_score {
            best_score = score;
            best_f0 = f0;
        }
    }

    best_f0
}

/// Detect F2 formant frequency by finding spectral centroid in F2 band.
///
/// Uses weighted centroid of spectral power in F2 band (1000-2500 Hz).
/// With wide formant bandwidth (BW2=300 Hz), energy spreads across harmonics,
/// making centroid more stable across different F0 values.
pub fn detect_formants(spectrum: &[f32], _f0: f64) -> f64 {
    let bin_lo = (F2_LO / HZ_PER_BIN).ceil() as usize;
    let bin_hi = (F2_HI / HZ_PER_BIN).floor() as usize;

    if bin_hi >= spectrum.len() || bin_lo >= bin_hi {
        return (F2_LO + F2_HI) / 2.0; // Fallback to center
    }

    // Find spectral centroid (power-weighted frequency average)
    let mut power_sum = 0.0f64;
    let mut weighted_sum = 0.0f64;

    for bin in bin_lo..=bin_hi.min(spectrum.len() - 1) {
        let freq = bin as f64 * HZ_PER_BIN;
        let power = spectrum[bin] as f64;
        power_sum += power;
        weighted_sum += freq * power;
    }

    if power_sum > 0.0 {
        weighted_sum / power_sum
    } else {
        (F2_LO + F2_HI) / 2.0
    }
}

/// Detect F1 and F2 formants using harmonic peak detection.
///
/// This is the secondary detection path for hybrid mode. It finds the
/// strongest harmonic in each formant band, which works well on clean
/// channels but may fail when F1 is crushed by VoIP processing.
pub fn detect_formants_f1f2(spectrum: &[f32], f0: f64) -> (f64, f64) {
    let mut best_f1_power = 0.0f64;
    let mut best_f1_freq = (F1_LO + F1_HI) / 2.0;
    let mut best_f2_power = 0.0f64;
    let mut best_f2_freq = (F2_LO + F2_HI) / 2.0;

    for h in 1..=NUM_HARMONICS {
        let freq = f0 * h as f64;
        let bin = (freq / HZ_PER_BIN).round() as usize;
        if bin >= spectrum.len() {
            break;
        }

        // Sum power in 3-bin window around harmonic
        let mut power = 0.0f64;
        for offset in 0..=2usize {
            if offset == 0 {
                power += spectrum[bin] as f64;
            } else {
                if bin >= offset && bin - offset < spectrum.len() {
                    power += spectrum[bin - offset] as f64;
                }
                if bin + offset < spectrum.len() {
                    power += spectrum[bin + offset] as f64;
                }
            }
        }

        if freq >= F1_LO && freq <= F1_HI && power > best_f1_power {
            best_f1_power = power;
            best_f1_freq = freq;
        }
        if freq >= F2_LO && freq <= F2_HI && power > best_f2_power {
            best_f2_power = power;
            best_f2_freq = freq;
        }
    }

    (best_f1_freq, best_f2_freq)
}

/// Classify detected F2 to the nearest vowel (1D classification).
///
/// Uses simple distance in F2. Returns (vowel_index, margin_confidence).
/// This is the primary classifier, robust to VoIP F1 crushing.
pub fn classify_vowel_f2(f2: f64) -> (usize, f64) {
    let mut best_idx = 0;
    let mut best_dist = f64::MAX;
    let mut second_dist = f64::MAX;

    for (i, v) in VOWELS.iter().enumerate() {
        let dist = (f2 - v.f2).abs() / 200.0; // Normalize by spacing

        if dist < best_dist {
            second_dist = best_dist;
            best_dist = dist;
            best_idx = i;
        } else if dist < second_dist {
            second_dist = dist;
        }
    }

    let confidence = if best_dist > 0.0 {
        (second_dist - best_dist) / best_dist
    } else {
        second_dist
    };

    (best_idx, confidence)
}

/// Classify detected (F1, F2) to the nearest vowel (2D classification).
///
/// Uses Euclidean distance in normalized F1/F2 space.
/// This is the secondary classifier for hybrid mode, used when F1 is available.
pub fn classify_vowel_f1f2(f1: f64, f2: f64) -> (usize, f64) {
    let mut best_idx = 0;
    let mut best_dist = f64::MAX;
    let mut second_dist = f64::MAX;

    for (i, v) in VOWELS.iter().enumerate() {
        let d1 = (f1 - v.f1) / 100.0; // F1 spacing ~100 Hz
        let d2 = (f2 - v.f2) / 200.0; // F2 spacing ~200 Hz
        let dist = (d1 * d1 + d2 * d2).sqrt();

        if dist < best_dist {
            second_dist = best_dist;
            best_dist = dist;
            best_idx = i;
        } else if dist < second_dist {
            second_dist = dist;
        }
    }

    let confidence = if best_dist > 0.0 {
        (second_dist - best_dist) / best_dist
    } else {
        second_dist
    };

    (best_idx, confidence)
}

/// Detect pitch class from detected F0.
pub fn detect_pitch(f0: f64) -> (PitchClass, f64) {
    let margin = (f0 - PITCH_THRESHOLD).abs();
    if f0 < PITCH_THRESHOLD {
        (PitchClass::Low, margin)
    } else {
        (PitchClass::High, margin)
    }
}

/// Classify a full symbol from a power spectrum using hybrid detection.
///
/// Runs two parallel detection paths:
/// 1. F2-only (primary): robust to VoIP F1 crushing
/// 2. F1+F2 (secondary): better accuracy on clean channels
///
/// Picks the result with higher confidence, combining the best of both.
pub fn classify_symbol(spectrum: &[f32]) -> (usize, f64) {
    let f0 = detect_f0(spectrum);
    let (pitch_class, pitch_conf) = detect_pitch(f0);
    let pitch_conf_norm = normalize_pitch_confidence(pitch_conf);

    // Path 1: F2-only detection (VoIP-safe, primary)
    let f2_centroid = detect_formants(spectrum, f0);
    let (vowel_f2, conf_f2) = classify_vowel_f2(f2_centroid);
    let conf_f2_norm = normalize_confidence(conf_f2);

    // Path 2: F1+F2 detection (secondary, for clean channels)
    let (f1_peak, f2_peak) = detect_formants_f1f2(spectrum, f0);
    let (vowel_f1f2, conf_f1f2) = classify_vowel_f1f2(f1_peak, f2_peak);
    let conf_f1f2_norm = normalize_confidence(conf_f1f2);

    // Hybrid decision logic:
    // - If both paths agree: use result with boosted confidence
    // - If they disagree: prefer F2-only (more reliable under distortion)
    //   unless F1+F2 confidence is significantly higher
    let (vowel_idx, vowel_conf) = if vowel_f2 == vowel_f1f2 {
        // Agreement: boost confidence
        (vowel_f2, (conf_f2_norm + conf_f1f2_norm) / 2.0 + 0.1)
    } else if conf_f1f2_norm > conf_f2_norm + 0.15 {
        // F1+F2 has significantly higher confidence, trust it
        (vowel_f1f2, conf_f1f2_norm)
    } else {
        // Disagreement: prefer F2-only (VoIP-safe default)
        (vowel_f2, conf_f2_norm)
    };

    let symbol_idx = params_to_symbol(vowel_idx, pitch_class as usize);

    // Composite confidence weighted by channel importance.
    let confidence = vowel_conf.min(1.0) * 0.7 + pitch_conf_norm * 0.3;

    (symbol_idx, confidence)
}

/// Map raw vowel confidence (0..∞) into 0..1 for stable mixing.
#[inline]
fn normalize_confidence(raw: f64) -> f64 {
    let raw = raw.max(0.0);
    raw / (raw + 1.0)
}

/// Map pitch margin (Hz) into 0..1 based on candidate extremes.
#[inline]
fn normalize_pitch_confidence(margin_hz: f64) -> f64 {
    // Candidates span 200..280 Hz with threshold at 240 => max margin = 40 Hz.
    const MAX_MARGIN_HZ: f64 = 40.0;
    (margin_hz / MAX_MARGIN_HZ).clamp(0.0, 1.0)
}

// --- 4-bit packing ---

/// Pack bytes into 4-bit symbol indices.
///
/// Converts a byte slice into a bitstream, then extracts 4-bit chunks.
/// Pads with zeros at the end if not aligned.
pub fn bytes_to_symbols(bytes: &[u8]) -> Vec<usize> {
    let total_bits = bytes.len() * 8;
    let n_symbols = (total_bits + BITS_PER_SYMBOL - 1) / BITS_PER_SYMBOL;
    let mut symbols = Vec::with_capacity(n_symbols);

    let mut bit_pos = 0;
    for _ in 0..n_symbols {
        let mut val = 0usize;
        for b in 0..BITS_PER_SYMBOL {
            let byte_idx = (bit_pos + b) / 8;
            let bit_idx = 7 - ((bit_pos + b) % 8); // MSB first
            if byte_idx < bytes.len() {
                val |= (((bytes[byte_idx] >> bit_idx) & 1) as usize) << (BITS_PER_SYMBOL - 1 - b);
            }
        }
        symbols.push(val);
        bit_pos += BITS_PER_SYMBOL;
    }

    symbols
}

/// Unpack 4-bit symbol indices back into bytes.
///
/// Extracts the bitstream from symbol indices and packs into bytes.
/// `num_bytes` specifies how many output bytes to produce.
pub fn symbols_to_bytes(symbols: &[usize], num_bytes: usize) -> Vec<u8> {
    let mut bytes = vec![0u8; num_bytes];

    let mut bit_pos = 0;
    for &sym in symbols {
        for b in 0..BITS_PER_SYMBOL {
            let byte_idx = (bit_pos + b) / 8;
            let bit_idx = 7 - ((bit_pos + b) % 8);
            if byte_idx < num_bytes {
                let bit_val = (sym >> (BITS_PER_SYMBOL - 1 - b)) & 1;
                bytes[byte_idx] |= (bit_val as u8) << bit_idx;
            }
        }
        bit_pos += BITS_PER_SYMBOL;
    }

    bytes
}

/// Compute number of symbols needed to encode n_bytes.
pub fn symbols_for_bytes(n_bytes: usize) -> usize {
    (n_bytes * 8 + BITS_PER_SYMBOL - 1) / BITS_PER_SYMBOL
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_symbol_decompose_compose_roundtrip() {
        for idx in 0..NUM_SYMBOLS {
            let (v, p) = symbol_to_params(idx);
            let back = params_to_symbol(v, p as usize);
            assert_eq!(idx, back, "symbol {idx} decompose/compose mismatch");
        }
    }

    #[test]
    fn test_symbol_vowel_extraction() {
        for idx in 0..NUM_SYMBOLS {
            let (v, _) = symbol_to_params(idx);
            assert_eq!(v, symbol_vowel(idx));
        }
    }

    #[test]
    fn test_bit_packing_roundtrip() {
        for len in 1..=20 {
            let bytes: Vec<u8> = (0..len).map(|i| ((i * 37 + 13) % 256) as u8).collect();
            let symbols = bytes_to_symbols(&bytes);
            let back = symbols_to_bytes(&symbols, len);
            assert_eq!(bytes, back, "packing roundtrip failed for len {len}");
        }
    }

    #[test]
    fn test_bit_packing_all_byte_values() {
        for b in 0..=255u8 {
            let bytes = vec![b];
            let symbols = bytes_to_symbols(&bytes);
            let back = symbols_to_bytes(&symbols, 1);
            assert_eq!(bytes, back, "packing roundtrip failed for byte {b:#04x}");
        }
    }

    #[test]
    fn test_bit_packing_known_values() {
        // All zeros
        let syms = bytes_to_symbols(&[0, 0, 0]);
        assert!(syms.iter().all(|&s| s == 0));

        // All ones: 0xFF, 0xFF, 0xFF = 24 bits = eight 3-bit symbols of 7
        let syms = bytes_to_symbols(&[0xFF, 0xFF, 0xFF]);
        assert_eq!(syms.len(), 8);
        assert!(syms.iter().all(|&s| s == 7));
    }

    #[test]
    fn test_harmonic_amplitudes_shape() {
        // F2=2080 at F0_LOW=208 Hz → H10=2080 should be strong
        // Use F1=500 as a dummy value (we're testing F2 behavior)
        let amps = harmonic_amplitudes(500.0, 2080.0, F0_LOW);
        assert!(amps[9] > 0.5, "H10 should be strong near F2=2080: {}", amps[9]);
    }

    #[test]
    fn test_classify_known_vowels() {
        for (i, v) in VOWELS.iter().enumerate() {
            let (sym, conf) = classify_vowel_f2(v.f2);
            assert_eq!(sym, i, "exact F2 should classify correctly for vowel {i}");
            assert!(conf > 0.0, "confidence should be positive for vowel {i}");
        }
    }

    #[test]
    fn test_symbols_for_bytes() {
        assert_eq!(symbols_for_bytes(1), 3);  // 8 bits → 3 symbols (9 bits capacity)
        assert_eq!(symbols_for_bytes(3), 8);  // 24 bits → 8 symbols (24 bits capacity)
        assert_eq!(symbols_for_bytes(6), 16); // 48 bits → 16 symbols (48 bits capacity)
    }
}
