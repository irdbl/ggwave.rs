# ggwave

Rust port of [ggwave](https://github.com/ggerganov/ggwave)'s **AUDIBLE_FAST** codec. Encode data into audible sound and decode it back — at 48 kHz, using frequency-shift keying with Reed-Solomon error correction.

## Usage

```rust
// Encode
let audio = ggwave::encode(b"hello", 25).unwrap(); // -> Vec<f32> at 48 kHz

// Decode
let mut decoder = ggwave::Decoder::new();
let payload = decoder.decode(&audio).unwrap().unwrap();
assert_eq!(&payload, b"hello");
```

The decoder is streaming — feed it audio in arbitrary-sized chunks and it returns `Ok(Some(data))` when a complete message is received:

```rust
let mut decoder = ggwave::Decoder::new();
for chunk in audio.chunks(512) {
    if let Ok(Some(payload)) = decoder.decode(chunk) {
        println!("{}", String::from_utf8_lossy(&payload));
    }
}
```

## API

### `encode(payload: &[u8], volume: u8) -> Result<Vec<f32>, Error>`

Encodes up to 140 bytes into f32 audio samples at 48 kHz. Volume range is 0–100.

### `Decoder`

- `Decoder::new()` — create a new decoder
- `decoder.decode(&[f32])` — feed samples, returns `Ok(Some(payload))` on success
- `decoder.reset()` — reset state for a new message

## Robustness

The decoder uses spectral confidence scoring to select the best decode candidate across sub-frame offsets, making it resilient to:

- 16-bit WAV quantization
- Additive noise (tolerates ~5% at volume 50)
- Amplitude clipping and scaling
- DC offset
- Leading/trailing silence or noise
- Degenerate payloads (all-zeros, repeated bytes)

All 256 single-byte values and all payload lengths 1–140 round-trip correctly.

## Protocol details

| Parameter | Value |
|---|---|
| Sample rate | 48,000 Hz |
| Samples per frame | 1,024 |
| Max payload | 140 bytes |
| Error correction | Reed-Solomon (adaptive ECC) |
| Marker frames | 16 start + 16 end |
| Frequency base | 1,875 Hz (bin 40) |
| Frequency spacing | 93.75 Hz |

## Benchmarks

Run with `cargo bench`.

### Protocol bandwidth

Over-the-air data rate, computed from protocol parameters (not a runtime measurement):

| Payload | ECC | Frames | Duration | Bitrate |
|--------:|----:|-------:|---------:|--------:|
| 1 B | 2 B | 44 | 0.94 s | 8.5 bit/s |
| 5 B | 4 B | 56 | 1.19 s | 33.5 bit/s |
| 10 B | 4 B | 68 | 1.45 s | 55.1 bit/s |
| 25 B | 10 B | 110 | 2.35 s | 85.2 bit/s |
| 50 B | 20 B | 182 | 3.88 s | 103.0 bit/s |
| 100 B | 40 B | 320 | 6.83 s | 117.2 bit/s |
| 140 B | 56 B | 434 | 9.26 s | 121.0 bit/s |

Larger payloads amortize the fixed 32 marker frames, reaching ~121 bit/s at max size.

### Computational throughput

Encode and decode speed relative to the audio duration they produce/consume (Apple M2):

| Payload | Encode | Decode | Roundtrip |
|--------:|-------:|-------:|----------:|
| 1 B | 1.8 ms (507x RT) | 7.1 ms (132x RT) | 8.8 ms |
| 5 B | 1.8 ms (641x RT) | 2.7 ms (437x RT) | 4.5 ms |
| 10 B | 1.9 ms (766x RT) | 4.0 ms (356x RT) | 5.8 ms |
| 25 B | 2.0 ms (1138x RT) | 7.7 ms (304x RT) | 9.7 ms |
| 50 B | 2.3 ms (1659x RT) | 14.3 ms (268x RT) | 16.6 ms |
| 100 B | 2.7 ms (2505x RT) | 28.2 ms (242x RT) | 30.7 ms |
| 140 B | 3.0 ms (3004x RT) | 41.6 ms (221x RT) | 44.4 ms |

Both encode and decode run >100x faster than real-time at all payload sizes.

## Dependencies

- [`rustfft`](https://crates.io/crates/rustfft) — FFT computation
- [`thiserror`](https://crates.io/crates/thiserror) — error types
- [`hound`](https://crates.io/crates/hound) — WAV I/O (dev only, for tests)
- [`criterion`](https://crates.io/crates/criterion) — benchmarking (dev only)

## License

MIT
