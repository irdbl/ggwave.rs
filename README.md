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

## Dependencies

- [`rustfft`](https://crates.io/crates/rustfft) — FFT computation
- [`thiserror`](https://crates.io/crates/thiserror) — error types
- [`hound`](https://crates.io/crates/hound) — WAV I/O (dev only, for tests)

## License

MIT
