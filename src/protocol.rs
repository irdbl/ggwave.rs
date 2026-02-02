/// Sample rate in Hz.
#[cfg(not(feature = "8khz"))]
pub const SAMPLE_RATE: f64 = 48000.0;
#[cfg(feature = "8khz")]
pub const SAMPLE_RATE: f64 = 8000.0;

/// Low fundamental frequency (G#3 = 208 Hz).
pub const F0_LOW: f64 = 208.0;

/// High fundamental frequency (C#4 = 277 Hz, perfect fourth above G#3).
pub const F0_HIGH: f64 = 277.0;

/// Pitch threshold: below = low, above = high (midpoint of 208/277).
pub const PITCH_THRESHOLD: f64 = 242.5;

/// Number of harmonics in the synthesis model (H1=F0 .. H16=16*F0).
/// At 8 kHz (Nyquist = 4 kHz), we use 14 harmonics to stay in band.
/// (F0_HIGH=277Hz × 14 = 3878 Hz, safely under 4 kHz Nyquist)
#[cfg(not(feature = "8khz"))]
pub const NUM_HARMONICS: usize = 16;
#[cfg(feature = "8khz")]
pub const NUM_HARMONICS: usize = 14;

/// Samples per symbol (50 ms voiced duration).
#[cfg(not(feature = "8khz"))]
pub const SAMPLES_PER_SYMBOL: usize = 2400; // 50 ms at 48 kHz
#[cfg(feature = "8khz")]
pub const SAMPLES_PER_SYMBOL: usize = 400; // 50 ms at 8 kHz

/// Guard silence after each symbol (10 ms).
#[cfg(not(feature = "8khz"))]
pub const GUARD_SAMPLES: usize = 480; // 10 ms at 48 kHz
#[cfg(feature = "8khz")]
pub const GUARD_SAMPLES: usize = 80; // 10 ms at 8 kHz

/// Total samples per symbol slot (symbol + guard).
/// 2880 at 48 kHz, 480 at 8 kHz.
pub const SYMBOL_TOTAL_SAMPLES: usize = SAMPLES_PER_SYMBOL + GUARD_SAMPLES;

/// Number of vowel shapes in the alphabet.
/// Reduced to 4 for VoIP compatibility (F0_HIGH has only 6 harmonics in F2 band).
pub const NUM_VOWELS: usize = 4;

/// Number of pitch classes.
pub const NUM_PITCHES: usize = 2;

/// Total symbol alphabet size (4 * 2 = 8).
pub const NUM_SYMBOLS: usize = NUM_VOWELS * NUM_PITCHES;

/// Bits per symbol (log2(8) = 3).
pub const BITS_PER_SYMBOL: usize = 3;

/// Preamble length in symbols.
pub const PREAMBLE_LEN: usize = 4;

/// FFT size for spectral analysis (same as symbol length).
pub const FFT_SIZE: usize = SAMPLES_PER_SYMBOL;

/// Hz per FFT bin = SAMPLE_RATE / FFT_SIZE.
pub const HZ_PER_BIN: f64 = SAMPLE_RATE / FFT_SIZE as f64;

/// Formant bandwidth for F1 (Hz) — Gaussian sigma.
/// (Unused in F2-only mode, kept for compatibility)
pub const BW1: f64 = 80.0;

/// Formant bandwidth for F2 (Hz) — Gaussian sigma.
/// Very wide bandwidth (300 Hz) helps with F0-independent detection
/// by spreading energy across multiple harmonics.
pub const BW2: f64 = 300.0;

/// F1 band lower bound (Hz).
/// (Unused in F2-only mode — VoIP crushes this band)
pub const F1_LO: f64 = 300.0;

/// F1 band upper bound (Hz).
/// (Unused in F2-only mode)
pub const F1_HI: f64 = 850.0;

/// F2 band lower bound (Hz) — above VoIP processing.
pub const F2_LO: f64 = 1000.0;

/// F2 band upper bound (Hz).
pub const F2_HI: f64 = 2500.0;

/// Candidate F0 values for multi-F0 detection in decoder (3 per pitch cluster).
pub const F0_CANDIDATES: [f64; 6] = [198.0, 208.0, 218.0, 267.0, 277.0, 287.0];

/// Encoded data offset (1 byte length + 2 bytes ECC for length).
pub const ENCODED_DATA_OFFSET: usize = 3;

/// Maximum variable-length payload size.
pub const MAX_LENGTH_VARIABLE: usize = 140;

/// Maximum total encoded data size.
pub const MAX_DATA_SIZE: usize = 256;

/// Compute ECC byte count for a given payload length.
///
/// Matches C++: `len < 4 ? 2 : max(4, 2*(len/5))`
pub fn ecc_bytes_for_length(len: usize) -> usize {
    if len < 4 {
        2
    } else {
        4usize.max(2 * (len / 5))
    }
}
