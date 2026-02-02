use std::env;

fn main() {
    let text: String = env::args().skip(1).collect::<Vec<_>>().join(" ");
    let text = if text.is_empty() {
        "Hello from the vocal modem!".to_string()
    } else {
        text
    };

    // Encode (max 140 bytes, split if needed)
    let bytes = text.as_bytes();
    let chunks: Vec<&[u8]> = bytes.chunks(140).collect();

    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: 48000,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };

    let path = "/tmp/song_rs_play.wav";
    let mut writer = hound::WavWriter::create(path, spec).unwrap();

    // Half second of silence at the start
    for _ in 0..24000 {
        writer.write_sample(0i16).unwrap();
    }

    for (i, chunk) in chunks.iter().enumerate() {
        let audio = song_rs::encode(chunk, 50).unwrap();
        for &s in &audio {
            let val = (s * 32767.0).clamp(-32768.0, 32767.0) as i16;
            writer.write_sample(val).unwrap();
        }
        eprintln!(
            "Chunk {}/{}: {} bytes, {:.2}s audio",
            i + 1,
            chunks.len(),
            chunk.len(),
            audio.len() as f64 / 48000.0,
        );
        // Half second gap between chunks
        if i + 1 < chunks.len() {
            for _ in 0..24000 {
                writer.write_sample(0i16).unwrap();
            }
        }
    }

    // Half second of silence at the end
    for _ in 0..24000 {
        writer.write_sample(0i16).unwrap();
    }

    writer.finalize().unwrap();

    let total_samples = std::fs::metadata(path).unwrap().len() / 2; // rough
    eprintln!("Wrote {path} ({:.2}s)", total_samples as f64 / 48000.0);
}
