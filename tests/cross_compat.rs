//! 2-channel vocal modem tests: WAV generation and self round-trip.

use song_rs::Decoder;

/// Write f32 samples to a 48kHz WAV file.
fn write_wav(path: &str, samples: &[f32]) {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: 48000,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(path, spec).expect("failed to create WAV");
    for &s in samples {
        let val = (s * 32767.0).clamp(-32768.0, 32767.0) as i16;
        writer.write_sample(val).expect("write failed");
    }
    writer.finalize().expect("finalize failed");
}

/// Load a WAV file and return f32 samples.
fn load_wav(path: &str) -> Vec<f32> {
    let reader = hound::WavReader::open(path).expect("failed to open WAV");
    let spec = reader.spec();
    assert_eq!(spec.sample_rate, 48000, "expected 48kHz WAV");

    match spec.sample_format {
        hound::SampleFormat::Float => reader
            .into_samples::<f32>()
            .map(|s| s.expect("bad sample"))
            .collect(),
        hound::SampleFormat::Int => {
            let bits = spec.bits_per_sample;
            let max_val = (1i32 << (bits - 1)) as f32;
            reader
                .into_samples::<i32>()
                .map(|s| s.expect("bad sample") as f32 / max_val)
                .collect()
        }
    }
}

/// Generate voice WAV files and verify self round-trip through WAV (16-bit quantization).
#[test]
fn test_generate_voice_wavs() {
    let fox = b"The quick brown fox jumps over the lazy dog".to_vec();

    let cases: Vec<(&[u8], &str)> = vec![
        (b"hello", "/tmp/song_rs_hello.wav"),
        (b"a0Z5kR2g", "/tmp/song_rs_a0z5.wav"),
        (b"A", "/tmp/song_rs_single.wav"),
        (&fox, "/tmp/song_rs_fox.wav"),
    ];

    for (payload, path) in &cases {
        let audio = song_rs::encode(payload, 50).unwrap();
        write_wav(path, &audio);
        eprintln!("Wrote {path}: {} samples ({:.2}s)", audio.len(), audio.len() as f64 / 48000.0);

        // Verify we can decode our own WAV (tests 16-bit quantization robustness)
        let loaded = load_wav(path);
        let mut decoder = Decoder::new();
        let result = decoder.decode(&loaded).unwrap();
        assert_eq!(
            result.as_deref(),
            Some(*payload),
            "self round-trip via WAV failed for {:?}",
            String::from_utf8_lossy(payload)
        );
    }
}

/// Test all 256 single byte values round-trip.
#[test]
fn test_single_byte_all_values() {
    let mut failures = Vec::new();
    for b in 0..=255u8 {
        let payload = [b];
        let audio = song_rs::encode(&payload, 50).unwrap();
        let mut decoder = Decoder::new();
        match decoder.decode(&audio) {
            Ok(Some(data)) if data == vec![b] => {}
            Ok(Some(data)) => failures.push((b, format!("wrong: {:?}", data))),
            Ok(None) => failures.push((b, "None".into())),
            Err(e) => failures.push((b, format!("Err: {e}"))),
        }
    }
    if !failures.is_empty() {
        for (b, msg) in &failures {
            eprintln!("FAIL byte 0x{b:02X}: {msg}");
        }
        panic!("{} of 256 single-byte values failed", failures.len());
    }
}

/// Streaming decode test.
#[test]
fn test_streaming_voice_decode() {
    let cases: &[&[u8]] = &[b"hello", b"A", b"test data"];

    for &payload in cases {
        let audio = song_rs::encode(payload, 50).unwrap();
        let mut decoder = Decoder::new();
        let mut result = None;
        for chunk in audio.chunks(256) {
            if let Ok(Some(data)) = decoder.decode(chunk) {
                result = Some(data);
                break;
            }
        }
        assert_eq!(
            result.as_deref(),
            Some(payload),
            "streaming decode failed for {:?}",
            String::from_utf8_lossy(payload)
        );
    }
}
