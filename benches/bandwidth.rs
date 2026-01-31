use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use ggwave::protocol;

const PAYLOAD_SIZES: &[usize] = &[1, 5, 10, 25, 50, 100, 140];

/// Compute the over-the-air duration for a given payload size.
fn audio_duration_secs(payload_len: usize) -> f64 {
    let n_ecc = protocol::ecc_bytes_for_length(payload_len);
    let total_bytes = protocol::ENCODED_DATA_OFFSET + payload_len + n_ecc;
    let total_data_frames = protocol::EXTRA
        * ((total_bytes + protocol::BYTES_PER_TX - 1) / protocol::BYTES_PER_TX)
        * protocol::FRAMES_PER_TX;
    let total_frames =
        protocol::N_MARKER_FRAMES + total_data_frames + protocol::N_MARKER_FRAMES;
    let total_samples = total_frames * protocol::SAMPLES_PER_FRAME;
    total_samples as f64 / protocol::SAMPLE_RATE as f64
}

/// Print the protocol bandwidth table once before benchmarks run.
fn print_protocol_table() {
    println!();
    println!("=== Protocol Bandwidth (AUDIBLE_FAST @ 48 kHz) ===");
    println!(
        "{:>7} {:>5} {:>8} {:>10} {:>8} {:>12}",
        "Payload", "ECC", "Frames", "Samples", "Duration", "Bitrate"
    );
    println!(
        "{:>7} {:>5} {:>8} {:>10} {:>8} {:>12}",
        "(bytes)", "(bytes)", "", "", "(sec)", "(bit/s)"
    );
    println!("{}", "-".repeat(58));
    for &size in PAYLOAD_SIZES {
        let n_ecc = protocol::ecc_bytes_for_length(size);
        let total_bytes = protocol::ENCODED_DATA_OFFSET + size + n_ecc;
        let total_data_frames = protocol::EXTRA
            * ((total_bytes + protocol::BYTES_PER_TX - 1) / protocol::BYTES_PER_TX)
            * protocol::FRAMES_PER_TX;
        let total_frames =
            protocol::N_MARKER_FRAMES + total_data_frames + protocol::N_MARKER_FRAMES;
        let total_samples = total_frames * protocol::SAMPLES_PER_FRAME;
        let duration = total_samples as f64 / protocol::SAMPLE_RATE as f64;
        let payload_bits = size * 8;
        let bitrate = payload_bits as f64 / duration;
        println!(
            "{:>7} {:>5} {:>8} {:>10} {:>8.2} {:>10.1}",
            size, n_ecc, total_frames, total_samples, duration, bitrate,
        );
    }
    println!();
}

fn make_payload(size: usize) -> Vec<u8> {
    (0..size).map(|i| ((i * 7 + 13) % 256) as u8).collect()
}

fn bench_encode(c: &mut Criterion) {
    print_protocol_table();

    let mut group = c.benchmark_group("encode");
    for &size in PAYLOAD_SIZES {
        let payload = make_payload(size);
        let duration = audio_duration_secs(size);
        group.throughput(Throughput::Bytes(size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), &payload, |b, payload| {
            b.iter(|| ggwave::encode(payload, 50).unwrap());
        });
        // Print real-time multiplier after each benchmark using rough estimate
        let encode_time = {
            let start = std::time::Instant::now();
            for _ in 0..10 {
                let _ = ggwave::encode(&payload, 50).unwrap();
            }
            start.elapsed().as_secs_f64() / 10.0
        };
        println!(
            "  encode/{size}: audio {duration:.2}s, encode {encode_time:.4}s -> {:.0}x real-time",
            duration / encode_time
        );
    }
    group.finish();
}

fn bench_decode(c: &mut Criterion) {
    let mut group = c.benchmark_group("decode");
    for &size in PAYLOAD_SIZES {
        let payload = make_payload(size);
        let audio = ggwave::encode(&payload, 50).unwrap();
        let duration = audio_duration_secs(size);
        group.throughput(Throughput::Bytes(size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), &audio, |b, audio| {
            b.iter(|| {
                let mut decoder = ggwave::Decoder::new();
                decoder.decode(audio).unwrap().unwrap();
            });
        });
        let decode_time = {
            let start = std::time::Instant::now();
            for _ in 0..10 {
                let mut decoder = ggwave::Decoder::new();
                decoder.decode(&audio).unwrap().unwrap();
            }
            start.elapsed().as_secs_f64() / 10.0
        };
        println!(
            "  decode/{size}: audio {duration:.2}s, decode {decode_time:.4}s -> {:.0}x real-time",
            duration / decode_time
        );
    }
    group.finish();
}

fn bench_roundtrip(c: &mut Criterion) {
    let mut group = c.benchmark_group("roundtrip");
    for &size in PAYLOAD_SIZES {
        let payload = make_payload(size);
        group.throughput(Throughput::Bytes(size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), &payload, |b, payload| {
            b.iter(|| {
                let audio = ggwave::encode(payload, 50).unwrap();
                let mut decoder = ggwave::Decoder::new();
                decoder.decode(&audio).unwrap().unwrap();
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_encode, bench_decode, bench_roundtrip);
criterion_main!(benches);
