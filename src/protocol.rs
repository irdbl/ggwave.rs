/// Sample rate in Hz.
pub const SAMPLE_RATE: f32 = 48000.0;

/// Number of samples per FFT frame.
pub const SAMPLES_PER_FRAME: usize = 1024;

/// Frequency resolution: SAMPLE_RATE / SAMPLES_PER_FRAME = 46.875 Hz per bin.
pub const HZ_PER_SAMPLE: f64 = SAMPLE_RATE as f64 / SAMPLES_PER_FRAME as f64;

/// Inverse of HZ_PER_SAMPLE.
pub const IHZ_PER_SAMPLE: f64 = 1.0 / HZ_PER_SAMPLE;

/// FFT bin index of the lowest frequency used by AUDIBLE_FAST.
pub const FREQ_START: usize = 40;

/// Number of frames transmitted per chunk of data.
pub const FRAMES_PER_TX: usize = 6;

/// Number of bytes per transmitted chunk.
pub const BYTES_PER_TX: usize = 3;

/// Extra factor (1 = standard, 2 = mono-tone). AUDIBLE protocols use 1.
pub const EXTRA: usize = 1;

/// Number of tones per transmission = (2 * BYTES_PER_TX) / EXTRA.
pub const N_TONES: usize = (2 * BYTES_PER_TX) / EXTRA;

/// Number of data bits per transmission = 8 * BYTES_PER_TX.
pub const N_DATA_BITS_PER_TX: usize = 8 * BYTES_PER_TX;

/// Bin spacing between bit0 and bit1 frequencies for a tone.
pub const FREQ_DELTA_BIN: usize = 1;

/// Hz spacing = 2 * HZ_PER_SAMPLE.
pub const FREQ_DELTA_HZ: f64 = 2.0 * HZ_PER_SAMPLE;

/// Number of bits used in marker detection.
pub const N_BITS_IN_MARKER: usize = 16;

/// Number of frames for start/end markers.
pub const N_MARKER_FRAMES: usize = 16;

/// Encoded data offset (1 byte length + 2 bytes ECC for length).
pub const ENCODED_DATA_OFFSET: usize = 3;

/// Sound marker detection threshold.
pub const SOUND_MARKER_THRESHOLD: f64 = 3.0;

/// Maximum variable-length payload size.
pub const MAX_LENGTH_VARIABLE: usize = 140;

/// Maximum total encoded data size.
pub const MAX_DATA_SIZE: usize = 256;

/// Maximum number of recorded frames for variable-length decoding.
pub const MAX_RECORDED_FRAMES: usize = 2048;

/// Number of spectrum history frames for averaging.
pub const MAX_SPECTRUM_HISTORY: usize = 4;

/// Compute the frequency (in Hz) for a given bit index within a protocol.
///
/// bit_freq = HZ_PER_SAMPLE * FREQ_START + FREQ_DELTA_HZ * bit
#[inline]
pub fn bit_freq(bit: usize) -> f64 {
    HZ_PER_SAMPLE * FREQ_START as f64 + FREQ_DELTA_HZ * bit as f64
}

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
