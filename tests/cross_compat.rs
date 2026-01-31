//! Cross-compatibility tests: decode C++-generated WAV files with Rust decoder,
//! and generate Rust WAV files for decoding with C++ tools.

use ggwave::Decoder;

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

#[test]
fn test_decode_cpp_hello() {
    let path = "/tmp/ggwave_hello_cpp.wav";
    if !std::path::Path::new(path).exists() {
        eprintln!("SKIP: {path} not found (generate with C++ ggwave-to-file)");
        return;
    }
    let samples = load_wav(path);
    eprintln!("Loaded {path}: {} samples", samples.len());

    let mut decoder = Decoder::new();
    let result = decoder.decode(&samples);
    match &result {
        Ok(Some(payload)) => {
            let text = String::from_utf8_lossy(payload);
            eprintln!("Decoded: {:?}", text);
            assert_eq!(payload, b"hello", "expected 'hello', got {:?}", text);
        }
        Ok(None) => {
            panic!("Decoder returned None — not enough data or marker not found");
        }
        Err(e) => {
            panic!("Decoder error: {e}");
        }
    }
}

#[test]
fn test_decode_cpp_a0z5() {
    let path = "/tmp/ggwave_a0z5_cpp.wav";
    if !std::path::Path::new(path).exists() {
        eprintln!("SKIP: {path} not found");
        return;
    }
    let samples = load_wav(path);
    eprintln!("Loaded {path}: {} samples", samples.len());

    let mut decoder = Decoder::new();
    let result = decoder.decode(&samples);
    match &result {
        Ok(Some(payload)) => {
            let text = String::from_utf8_lossy(payload);
            eprintln!("Decoded: {:?}", text);
            assert_eq!(payload, b"a0Z5kR2g", "expected 'a0Z5kR2g', got {:?}", text);
        }
        Ok(None) => panic!("Decoder returned None"),
        Err(e) => panic!("Decoder error: {e}"),
    }
}

#[test]
fn test_decode_cpp_single() {
    let path = "/tmp/ggwave_single_cpp.wav";
    if !std::path::Path::new(path).exists() {
        eprintln!("SKIP: {path} not found");
        return;
    }
    let samples = load_wav(path);
    eprintln!("Loaded {path}: {} samples", samples.len());

    let mut decoder = Decoder::new();
    let result = decoder.decode(&samples);
    match &result {
        Ok(Some(payload)) => {
            let text = String::from_utf8_lossy(payload);
            eprintln!("Decoded: {:?}", text);
            assert_eq!(payload, b"A", "expected 'A', got {:?}", text);
        }
        Ok(None) => panic!("Decoder returned None"),
        Err(e) => panic!("Decoder error: {e}"),
    }
}

/// Decode C++ WAV with medium-length message (44 bytes).
#[test]
fn test_decode_cpp_fox() {
    let path = "/tmp/ggwave_fox_cpp.wav";
    if !std::path::Path::new(path).exists() {
        eprintln!("SKIP: {path} not found");
        return;
    }
    let samples = load_wav(path);
    eprintln!("Loaded {path}: {} samples", samples.len());

    let mut decoder = Decoder::new();
    let result = decoder.decode(&samples);
    match &result {
        Ok(Some(payload)) => {
            let text = String::from_utf8_lossy(payload);
            eprintln!("Decoded: {:?}", text);
            assert_eq!(
                payload,
                b"The quick brown fox jumps over the lazy dog",
                "got {:?}",
                text
            );
        }
        Ok(None) => panic!("Decoder returned None"),
        Err(e) => panic!("Decoder error: {e}"),
    }
}

/// Decode C++ WAV generated at low volume (v=10).
#[test]
fn test_decode_cpp_quiet() {
    let path = "/tmp/ggwave_quiet_cpp.wav";
    if !std::path::Path::new(path).exists() {
        eprintln!("SKIP: {path} not found");
        return;
    }
    let samples = load_wav(path);
    eprintln!("Loaded {path}: {} samples", samples.len());

    let mut decoder = Decoder::new();
    let result = decoder.decode(&samples);
    match &result {
        Ok(Some(payload)) => {
            let text = String::from_utf8_lossy(payload);
            eprintln!("Decoded: {:?}", text);
            assert_eq!(payload, b"quiet", "got {:?}", text);
        }
        Ok(None) => panic!("Decoder returned None"),
        Err(e) => panic!("Decoder error: {e}"),
    }
}

/// Decode C++ WAV generated at max volume (v=100).
#[test]
fn test_decode_cpp_loud() {
    let path = "/tmp/ggwave_loud_cpp.wav";
    if !std::path::Path::new(path).exists() {
        eprintln!("SKIP: {path} not found");
        return;
    }
    let samples = load_wav(path);
    eprintln!("Loaded {path}: {} samples", samples.len());

    let mut decoder = Decoder::new();
    let result = decoder.decode(&samples);
    match &result {
        Ok(Some(payload)) => {
            let text = String::from_utf8_lossy(payload);
            eprintln!("Decoded: {:?}", text);
            assert_eq!(payload, b"loud", "got {:?}", text);
        }
        Ok(None) => panic!("Decoder returned None"),
        Err(e) => panic!("Decoder error: {e}"),
    }
}

/// Decode C++ WAV with 64-byte binary payload.
#[test]
fn test_decode_cpp_binary64() {
    let path = "/tmp/ggwave_binary64_cpp.wav";
    if !std::path::Path::new(path).exists() {
        eprintln!("SKIP: {path} not found");
        return;
    }
    let samples = load_wav(path);
    eprintln!("Loaded {path}: {} samples", samples.len());

    // Bytes 0x20..0x60 (printable ASCII, avoids newline which stops C++ stdin read)
    let expected: Vec<u8> = (0x20..0x60).collect();
    let mut decoder = Decoder::new();
    let result = decoder.decode(&samples);
    match &result {
        Ok(Some(payload)) => {
            eprintln!("Decoded {} bytes", payload.len());
            assert_eq!(payload, &expected, "binary payload mismatch");
        }
        Ok(None) => panic!("Decoder returned None"),
        Err(e) => panic!("Decoder error: {e}"),
    }
}

/// Decode C++ WAVs via streaming (small chunks) to stress the streaming decoder path.
#[test]
fn test_decode_cpp_streaming() {
    let cases: &[(&str, &[u8])] = &[
        ("/tmp/ggwave_hello_cpp.wav", b"hello"),
        ("/tmp/ggwave_single_cpp.wav", b"A"),
    ];

    for &(path, expected) in cases {
        if !std::path::Path::new(path).exists() {
            eprintln!("SKIP: {path} not found");
            continue;
        }
        let samples = load_wav(path);
        let mut decoder = Decoder::new();
        let mut result = None;
        // Feed 256 samples at a time (quarter of a frame)
        for chunk in samples.chunks(256) {
            if let Ok(Some(data)) = decoder.decode(chunk) {
                result = Some(data);
                break;
            }
        }
        assert_eq!(
            result.as_deref(),
            Some(expected),
            "streaming decode of {path} failed"
        );
    }
}

/// Decode C++ WAV with leading silence prepended (simulates real-world mic capture).
#[test]
fn test_decode_cpp_with_leading_silence() {
    let path = "/tmp/ggwave_hello_cpp.wav";
    if !std::path::Path::new(path).exists() {
        eprintln!("SKIP: {path} not found");
        return;
    }
    let samples = load_wav(path);

    // Prepend 0.5 seconds of silence (24000 samples)
    let silence = vec![0.0f32; 24000];
    let mut padded = silence;
    padded.extend_from_slice(&samples);

    let mut decoder = Decoder::new();
    let result = decoder.decode(&padded);
    match &result {
        Ok(Some(payload)) => {
            assert_eq!(payload, b"hello", "leading silence test failed");
        }
        Ok(None) => panic!("Decoder returned None with leading silence"),
        Err(e) => panic!("Decoder error with leading silence: {e}"),
    }
}

/// Generate Rust WAV files and verify self round-trip through WAV (16-bit quantization).
#[test]
fn test_generate_rust_wavs() {
    let fox = b"The quick brown fox jumps over the lazy dog".to_vec();
    let binary64: Vec<u8> = (0x20..0x60).collect();

    let cases: Vec<(&[u8], &str)> = vec![
        (b"hello", "/tmp/ggwave_hello_rust.wav"),
        (b"a0Z5kR2g", "/tmp/ggwave_a0z5_rust.wav"),
        (b"A", "/tmp/ggwave_single_rust.wav"),
        (&fox, "/tmp/ggwave_fox_rust.wav"),
        (&binary64, "/tmp/ggwave_binary64_rust.wav"),
    ];

    for (payload, path) in &cases {
        let audio = ggwave::encode(payload, 50).unwrap();
        write_wav(path, &audio);
        eprintln!("Wrote {path}: {} samples", audio.len());

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

/// Debug test: previously-failing byte values.
#[test]
fn test_debug_problematic_bytes() {
    let problem_bytes = [0x02u8, 0x03, 0x12, 0x32, 0x42, 0x62, 0x72, 0x73, 0x6D];
    for &b in &problem_bytes {
        let payload = [b];
        let audio = ggwave::encode(&payload, 50).unwrap();
        let mut decoder = Decoder::new();
        let result = decoder.decode(&audio).unwrap();
        assert_eq!(
            result.as_deref(),
            Some(&payload[..]),
            "round-trip failed for byte 0x{b:02X}: got {:?}",
            result
        );
    }
}

/// Test: which single byte values fail round-trip?
#[test]
fn test_single_byte_failures() {
    let mut failures = Vec::new();
    for b in 0..=255u8 {
        let payload = [b];
        let audio = ggwave::encode(&payload, 50).unwrap();
        let mut decoder = Decoder::new();
        match decoder.decode(&audio) {
            Ok(Some(data)) if data == vec![b] => {} // ok
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
