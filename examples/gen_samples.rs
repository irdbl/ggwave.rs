use ggwave_voice::{StreamTx, StreamConfig};

fn write_wav(path: &str, samples: &[f32]) {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: 48000,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut w = hound::WavWriter::create(path, spec).unwrap();
    for &s in samples {
        w.write_sample((s * 32767.0).clamp(-32768.0, 32767.0) as i16).unwrap();
    }
    w.finalize().unwrap();
}

fn silence(n: usize) -> Vec<f32> {
    vec![0.0f32; n]
}

/// Single-message encode with padding.
fn encode_single(text: &str, volume: u8) -> Vec<f32> {
    let mut audio = silence(24000); // 0.5s lead-in
    let encoded = ggwave_voice::encode(text.as_bytes(), volume).unwrap();
    audio.extend_from_slice(&encoded);
    audio.extend(silence(24000)); // 0.5s tail
    audio
}

/// Streaming encode with padding.
fn encode_stream(data: &[u8], config: StreamConfig) -> Vec<f32> {
    let mut tx = StreamTx::new(config);
    tx.feed(data);
    tx.finish();

    let mut audio = silence(24000); // 0.5s lead-in
    loop {
        let mut buf = vec![0.0f32; 2880];
        let n = tx.emit(&mut buf);
        if n == 0 { break; }
        audio.extend_from_slice(&buf[..n]);
    }
    audio.extend(silence(24000)); // 0.5s tail
    audio
}

fn main() {
    std::fs::create_dir_all("samples").unwrap();

    // Sample 1: Single message — the classic demo (~13s)
    let text1 = "The quick brown fox jumps over the lazy dog. This message was encoded as vowel sounds.";
    let audio1 = encode_single(text1, 50);
    write_wav("samples/single_message.wav", &audio1);
    eprintln!(
        "single_message.wav: {} bytes payload, {:.1}s audio",
        text1.len(),
        audio1.len() as f64 / 48000.0
    );

    // Sample 2: Streaming FEC — continuous transfer (~20s)
    let text2 = "Streaming mode sends a continuous RS-coded symbol stream. \
                 Reed-Solomon over GF(2^4) corrects burst errors.";
    let config = StreamConfig::default();
    let audio2 = encode_stream(text2.as_bytes(), config);
    write_wav("samples/streaming_fec.wav", &audio2);
    eprintln!(
        "streaming_fec.wav: {} bytes payload, {:.1}s audio",
        text2.len(),
        audio2.len() as f64 / 48000.0
    );

    // Sample 3: Max payload single message (~24s)
    let text3 = "Eight machine-optimized vowels carry three bits each \
                 via formant frequencies. Pitch adds one more bit per \
                 symbol, for sixteen total.";
    let audio3 = encode_single(text3, 50);
    write_wav("samples/max_payload.wav", &audio3);
    eprintln!(
        "max_payload.wav: {} bytes payload, {:.1}s audio",
        text3.len(),
        audio3.len() as f64 / 48000.0
    );

    eprintln!("Done. Convert to mp3: for f in samples/*.wav; do ffmpeg -i \"$f\" -b:a 128k \"${{f%.wav}}.mp3\" -y; done");
}
