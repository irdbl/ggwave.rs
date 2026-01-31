//! FFT wrapper around `rustfft` for computing power spectrum from real-valued audio.

use rustfft::{num_complex::Complex, FftPlanner};

use crate::protocol::SAMPLES_PER_FRAME;

/// Compute the power spectrum of a real-valued audio frame.
///
/// Input: `samples` of length `SAMPLES_PER_FRAME` (1024).
/// Output: `spectrum` of length `SAMPLES_PER_FRAME / 2 + 1` (513) containing
/// power (magnitude squared) at each bin.
///
/// The C++ implementation computes:
///   spectrum[i] = re[i]^2 + im[i]^2
/// and then folds the negative frequencies:
///   spectrum[i] += spectrum[N - i] for i in 1..N/2
///
/// We replicate that exactly.
pub fn power_spectrum(samples: &[f32], spectrum: &mut [f32]) {
    assert_eq!(samples.len(), SAMPLES_PER_FRAME);
    assert!(spectrum.len() >= SAMPLES_PER_FRAME);

    let n = SAMPLES_PER_FRAME;

    // Convert to complex
    let mut buffer: Vec<Complex<f32>> = samples
        .iter()
        .map(|&s| Complex::new(s, 0.0))
        .collect();

    // Perform FFT
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(n);
    fft.process(&mut buffer);

    // Compute power: |X[k]|^2
    for i in 0..n {
        spectrum[i] = buffer[i].norm_sqr();
    }

    // Fold negative frequencies: spectrum[i] += spectrum[N-i] for 1..N/2
    for i in 1..n / 2 {
        spectrum[i] += spectrum[n - i];
    }
}

/// Compute power spectrum from a windowed/averaged time-domain signal.
///
/// Same as `power_spectrum` but takes an arbitrary `&[f32]` slice and
/// writes to a pre-allocated buffer. Used by the decoder for sub-frame analysis.
pub fn power_spectrum_into(samples: &[f32], spectrum: &mut Vec<f32>) {
    let n = samples.len();
    spectrum.clear();
    spectrum.resize(n, 0.0);

    let mut buffer: Vec<Complex<f32>> = samples
        .iter()
        .map(|&s| Complex::new(s, 0.0))
        .collect();

    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(n);
    fft.process(&mut buffer);

    for i in 0..n {
        spectrum[i] = buffer[i].norm_sqr();
    }

    for i in 1..n / 2 {
        spectrum[i] += spectrum[n - i];
    }
}
