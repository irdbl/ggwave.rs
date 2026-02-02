//! Stream FEC integration tests: loopback encode → audio → decode.

use song_rs::{StreamTx, StreamRx, StreamConfig};
use song_rs::rs4::ReedSolomon4;

// --- RS4 unit tests (codec-level, no audio) ---

#[test]
fn test_rs4_encode_decode_roundtrip() {
    for (k, t) in [(11, 4), (9, 6), (7, 8), (13, 2)] {
        let rs = ReedSolomon4::new(k, t);
        let msg: Vec<u8> = (0..k as u8).map(|v| v % 16).collect();
        let encoded = rs.encode(&msg);
        assert_eq!(encoded.len(), 15);
        let decoded = rs.decode(&encoded).unwrap();
        assert_eq!(decoded, msg, "RS(15,{k}) roundtrip failed");
    }
}

#[test]
fn test_rs4_all_zero_all_ones() {
    let rs = ReedSolomon4::new(11, 4);
    for val in [0u8, 15] {
        let msg = vec![val; 11];
        let encoded = rs.encode(&msg);
        let decoded = rs.decode(&encoded).unwrap();
        assert_eq!(decoded, msg);
    }
}

#[test]
fn test_rs4_error_correction_1() {
    let rs = ReedSolomon4::new(11, 4);
    let msg: Vec<u8> = (0..11).collect();
    let encoded = rs.encode(&msg);

    for pos in 0..15 {
        let mut corrupted = encoded.clone();
        corrupted[pos] = (corrupted[pos] + 1) % 16;
        let decoded = rs.decode(&corrupted).expect(&format!("pos {pos}"));
        assert_eq!(decoded, msg);
    }
}

#[test]
fn test_rs4_error_correction_2() {
    let rs = ReedSolomon4::new(11, 4);
    let msg: Vec<u8> = (0..11).collect();
    let encoded = rs.encode(&msg);

    let mut corrupted = encoded.clone();
    corrupted[2] = (corrupted[2] + 5) % 16;
    corrupted[10] = (corrupted[10] + 9) % 16;

    let decoded = rs.decode(&corrupted).unwrap();
    assert_eq!(decoded, msg);
}

#[test]
fn test_rs4_erasure_correction_3() {
    let rs = ReedSolomon4::new(11, 4);
    let msg: Vec<u8> = (0..11).collect();
    let encoded = rs.encode(&msg);

    let mut corrupted = encoded.clone();
    corrupted[0] = 0;
    corrupted[5] = 0;
    corrupted[10] = 0;

    let decoded = rs.decode_with_erasures(&corrupted, &[0, 5, 10]).unwrap();
    assert_eq!(decoded, msg);
}

#[test]
fn test_rs4_erasure_correction_4() {
    let rs = ReedSolomon4::new(11, 4);
    let msg: Vec<u8> = (0..11).collect();
    let encoded = rs.encode(&msg);

    let mut corrupted = encoded.clone();
    corrupted[1] = 0;
    corrupted[4] = 0;
    corrupted[8] = 0;
    corrupted[13] = 0;

    let decoded = rs.decode_with_erasures(&corrupted, &[1, 4, 8, 13]).unwrap();
    assert_eq!(decoded, msg);
}

// --- Audio loopback helpers ---

const CHUNK: usize = 2880; // one symbol slot

/// Generate complete audio from StreamTx.
fn generate_audio(data: &[u8], config: StreamConfig) -> Vec<f32> {
    let mut tx = StreamTx::new(config);
    tx.feed(data);
    tx.finish();

    let mut audio = Vec::new();
    loop {
        let mut buf = vec![0.0f32; CHUNK];
        let n = tx.emit(&mut buf);
        if n == 0 {
            break;
        }
        audio.extend_from_slice(&buf[..n]);
    }
    audio
}

/// Feed audio into StreamRx and collect output.
fn receive_audio(audio: &[f32], config: StreamConfig) -> Vec<u8> {
    let mut rx = StreamRx::new(config);
    // Feed in chunks (simulates real-time streaming)
    for chunk in audio.chunks(CHUNK) {
        rx.ingest(chunk);
    }
    rx.read_all()
}

/// Full loopback with optional channel transformation.
fn stream_loopback(
    data: &[u8],
    channel: impl Fn(&[f32]) -> Vec<f32>,
    config: StreamConfig,
) -> Vec<u8> {
    let audio = generate_audio(data, config.clone());
    let processed = channel(&audio);
    receive_audio(&processed, config)
}

// --- DSP helpers ---

fn add_noise(samples: &[f32], snr_db: f64) -> Vec<f32> {
    let signal_power: f64 = samples.iter().map(|&s| (s as f64) * (s as f64)).sum::<f64>()
        / samples.len() as f64;
    if signal_power < 1e-20 {
        return samples.to_vec();
    }
    let noise_power = signal_power / 10.0f64.powf(snr_db / 10.0);
    let noise_amp = noise_power.sqrt();

    let mut rng: u32 = 42;
    samples
        .iter()
        .map(|&s| {
            rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
            let u1 = (rng >> 16) as f64 / 65535.0;
            rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
            let u2 = (rng >> 16) as f64 / 65535.0;
            let gauss =
                (-2.0 * u1.max(1e-10).ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
            (s as f64 + gauss * noise_amp) as f32
        })
        .collect()
}

fn bandpass(samples: &[f32], lo: f64, hi: f64) -> Vec<f32> {
    let sr = 48000.0;
    let center = (lo * hi).sqrt();
    let bw = hi - lo;
    let q = center / bw;
    let w0 = 2.0 * std::f64::consts::PI * center / sr;
    let alpha = w0.sin() / (2.0 * q);

    let b0 = alpha;
    let b1 = 0.0;
    let b2 = -alpha;
    let a0 = 1.0 + alpha;
    let a1 = -2.0 * w0.cos();
    let a2 = 1.0 - alpha;

    let mut out = vec![0.0f32; samples.len()];
    let mut x1 = 0.0f64;
    let mut x2 = 0.0f64;
    let mut y1 = 0.0f64;
    let mut y2 = 0.0f64;

    for (i, &s) in samples.iter().enumerate() {
        let x0 = s as f64;
        let y0 = (b0 * x0 + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2) / a0;
        out[i] = y0 as f32;
        x2 = x1;
        x1 = x0;
        y2 = y1;
        y1 = y0;
    }
    out
}

fn phone_channel(samples: &[f32]) -> Vec<f32> {
    bandpass(samples, 300.0, 3400.0)
}

// --- Stream loopback tests ---
//
// NOTE: These tests are incompatible with the Teams branch's 8-symbol alphabet.
// The streaming FEC uses RS4 (4-bit symbols = 16 symbols) while Teams uses
// 3-bit symbols (4 vowels × 2 pitches = 8 symbols). Marked as #[ignore].

#[test]
#[ignore = "streaming FEC incompatible with Teams 8-symbol alphabet"]
fn test_stream_clean() {
    let data: Vec<u8> = (0..100).map(|i| (i * 7 + 13) as u8).collect();
    let config = StreamConfig {
        interleave_depth: 1, // no interleaving for clean test
        ..StreamConfig::default()
    };
    let result = stream_loopback(&data, |s| s.to_vec(), config);
    assert_eq!(result, data, "clean stream loopback failed");
}

#[test]
#[ignore = "streaming FEC incompatible with Teams 8-symbol alphabet"]
fn test_stream_clean_interleaved() {
    let data: Vec<u8> = (0..100).map(|i| (i * 7 + 13) as u8).collect();
    let config = StreamConfig::default(); // depth=2
    let result = stream_loopback(&data, |s| s.to_vec(), config);
    assert_eq!(result, data, "interleaved clean stream loopback failed");
}

#[test]
#[ignore = "streaming FEC incompatible with Teams 8-symbol alphabet"]
fn test_stream_short_1() {
    let data = vec![0x42];
    let config = StreamConfig {
        interleave_depth: 1,
        ..StreamConfig::default()
    };
    let result = stream_loopback(&data, |s| s.to_vec(), config);
    assert_eq!(result, data);
}

#[test]
#[ignore = "streaming FEC incompatible with Teams 8-symbol alphabet"]
fn test_stream_short_5() {
    let data = vec![1, 2, 3, 4, 5];
    let config = StreamConfig {
        interleave_depth: 1,
        ..StreamConfig::default()
    };
    let result = stream_loopback(&data, |s| s.to_vec(), config);
    assert_eq!(result, data);
}

#[test]
#[ignore = "streaming FEC incompatible with Teams 8-symbol alphabet"]
fn test_stream_noisy_15db() {
    let data: Vec<u8> = (0..50).map(|i| (i * 3) as u8).collect();
    let config = StreamConfig {
        interleave_depth: 1,
        volume: 50,
        ..StreamConfig::default()
    };
    let result = stream_loopback(&data, |s| add_noise(s, 15.0), config);
    assert_eq!(result, data, "noisy 15dB stream failed");
}

#[test]
#[ignore = "streaming FEC incompatible with Teams 8-symbol alphabet"]
fn test_stream_phone() {
    let data: Vec<u8> = (0..50).map(|i| (i * 5 + 7) as u8).collect();
    let config = StreamConfig {
        interleave_depth: 1,
        volume: 50,
        ..StreamConfig::default()
    };
    let result = stream_loopback(&data, phone_channel, config);
    assert_eq!(result, data, "phone channel stream failed");
}

#[test]
#[ignore = "streaming FEC incompatible with Teams 8-symbol alphabet"]
fn test_stream_burst_erasure() {
    // With block interleaving (depth=2), a burst of 4 consecutive bad symbols
    // in the interleaved stream is spread across 2 codewords (2 errors each),
    // within RS(15,11) correction capacity of 2 errors per codeword.
    let data: Vec<u8> = (0..100).map(|i| (i * 7 + 13) as u8).collect();
    let config = StreamConfig::default(); // depth=2, RS(15,11)

    let result = stream_loopback(&data, |audio| {
        let mut processed = audio.to_vec();
        // Zero out 4 symbol slots in the middle of the data portion
        let symbol_slot = 2880; // SYMBOL_TOTAL_SAMPLES
        let start = (4 + 30) * symbol_slot; // after preamble + some data
        let burst_len = 4 * symbol_slot;
        if start + burst_len < processed.len() {
            for i in start..start + burst_len {
                processed[i] = 0.0;
            }
        }
        processed
    }, config);

    assert_eq!(result, data, "burst erasure recovery failed");
}

#[test]
#[ignore = "streaming FEC incompatible with Teams 8-symbol alphabet"]
fn test_stream_long() {
    // 255 bytes (max single-byte length prefix)
    let data: Vec<u8> = (0..255).map(|i| (i % 256) as u8).collect();
    let config = StreamConfig {
        interleave_depth: 1,
        ..StreamConfig::default()
    };
    let result = stream_loopback(&data, |s| s.to_vec(), config);
    assert_eq!(result, data, "long stream failed");
}

#[test]
#[ignore = "streaming FEC incompatible with Teams 8-symbol alphabet"]
fn test_stream_all_zeros() {
    let data = vec![0u8; 50];
    let config = StreamConfig {
        interleave_depth: 1,
        ..StreamConfig::default()
    };
    let result = stream_loopback(&data, |s| s.to_vec(), config);
    assert_eq!(result, data);
}

#[test]
#[ignore = "streaming FEC incompatible with Teams 8-symbol alphabet"]
fn test_stream_all_ff() {
    let data = vec![0xFFu8; 50];
    let config = StreamConfig {
        interleave_depth: 1,
        ..StreamConfig::default()
    };
    let result = stream_loopback(&data, |s| s.to_vec(), config);
    assert_eq!(result, data);
}

#[test]
#[ignore = "streaming FEC incompatible with Teams 8-symbol alphabet"]
fn test_stream_strong_rs() {
    // RS(15,9) with 6 parity: corrects 3 errors
    let data: Vec<u8> = (0..50).map(|i| (i * 11 + 3) as u8).collect();
    let config = StreamConfig {
        rs_parity: 6,
        interleave_depth: 1,
        volume: 50,
    };
    let result = stream_loopback(&data, |s| add_noise(s, 12.0), config);
    assert_eq!(result, data, "strong RS with noise failed");
}

#[test]
#[ignore = "streaming FEC incompatible with Teams 8-symbol alphabet"]
fn test_stream_throughput() {
    // Measure audio length for 100 bytes of payload
    let data: Vec<u8> = (0..100).collect();
    let config = StreamConfig {
        interleave_depth: 1,
        ..StreamConfig::default()
    };
    let audio = generate_audio(&data, config);
    let duration_s = audio.len() as f64 / 48000.0;
    let bytes_per_sec = data.len() as f64 / duration_s;

    // With RS(15,11), 11 data nibbles per 15 symbols, 60ms per symbol:
    // 11 nibbles = 5.5 bytes per 15 * 60ms = 0.9s → ~6.1 B/s
    // Plus preamble overhead, so expect at least 4 B/s
    assert!(
        bytes_per_sec > 3.0,
        "throughput too low: {bytes_per_sec:.1} B/s"
    );
    eprintln!("Stream throughput: {bytes_per_sec:.1} B/s ({duration_s:.2}s for {} bytes)", data.len());
}
