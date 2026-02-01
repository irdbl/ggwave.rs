# sóng

A voice-based acoustic modem that survives noise cancellation.

## The Problem

Acoustic modems like ggwave use pure sine tones. These get destroyed by phone call codecs (AMR, Opus), noise cancellation (Krisp, WebRTC), and speaker-air-microphone chains.

## The Insight

Noise cancellation preserves human voice. So we encode data as voice-like signals — formant positions instead of pure tones, harmonic structure instead of sine waves. If it sounds like speech, it survives.

## How It Works

Data is encoded as a sequence of **vowel-like sounds** using two orthogonal channels:

- **Vowel channel** (3 bits): F1/F2 formant frequencies select one of 8 vowels
- **Pitch channel** (1 bit): F0 fundamental frequency — 210 Hz (low) or 270 Hz (high)

Each symbol carries 4 bits (16 symbols total), synthesized as a harmonic series (16 harmonics) shaped by Gaussian formant envelopes.

8 machine-optimized vowels (maximizing classification distance across both F0 values):

| Vowel | F1 (Hz) | F2 (Hz) |
|-------|---------|---------|
| 0 | 480 | 1065 |
| 1 | 480 | 1890 |
| 2 | 480 | 2370 |
| 3 | 585 | 1410 |
| 4 | 720 | 1065 |
| 5 | 720 | 1890 |
| 6 | 825 | 1410 |
| 7 | 825 | 2370 |

All F1 values are ≥480 Hz to survive 300 Hz phone highpass filters.

## Listen

What data sounds like when encoded as vowels:

<table>
<tr><td><b>Single message</b> (86 bytes, 16s)</td></tr>
<tr><td><i>"The quick brown fox jumps over the lazy dog. This message was encoded as vowel sounds."</i></td></tr>
<tr><td>

https://github.com/irdbl/ggwave.rs/raw/master/samples/single_message.mp3

</td></tr>
<tr><td><b>Streaming FEC</b> (106 bytes, 20s)</td></tr>
<tr><td><i>"Streaming mode sends a continuous RS-coded symbol stream. Reed-Solomon over GF(2^4) corrects burst errors."</i></td></tr>
<tr><td>

https://github.com/irdbl/ggwave.rs/raw/master/samples/streaming_fec.mp3

</td></tr>
<tr><td><b>Full payload</b> (132 bytes, 24s)</td></tr>
<tr><td><i>"Eight machine-optimized vowels carry three bits each via formant frequencies. Pitch adds one more bit per symbol, for sixteen total."</i></td></tr>
<tr><td>

https://github.com/irdbl/ggwave.rs/raw/master/samples/max_payload.mp3

</td></tr>
</table>

## Usage

### Single-message API

```rust
// Encode
let audio = ggwave_voice::encode(b"hello", 25).unwrap(); // Vec<f32> at 48 kHz

// Decode
let mut decoder = ggwave_voice::Decoder::new();
let payload = decoder.decode(&audio).unwrap().unwrap();
assert_eq!(&payload, b"hello");
```

The decoder is streaming — feed audio in arbitrary-sized chunks:

```rust
let mut decoder = ggwave_voice::Decoder::new();
for chunk in audio.chunks(512) {
    if let Ok(Some(payload)) = decoder.decode(chunk) {
        println!("{}", String::from_utf8_lossy(&payload));
    }
}
```

### Streaming FEC API

For continuous data transfer without per-message preamble overhead:

```rust
use ggwave_voice::{StreamTx, StreamRx, StreamConfig};

let config = StreamConfig::default(); // RS(15,11), interleave depth 2

// Transmit
let mut tx = StreamTx::new(config.clone());
tx.feed(b"streaming data here");
tx.finish();
let mut audio = vec![0.0f32; 4096];
while tx.emit(&mut audio) > 0 {
    // write audio to speaker
}

// Receive
let mut rx = StreamRx::new(config);
rx.ingest(&audio);
let decoded = rx.read_all();
```

### PHY layer (reliable delivery)

For reliable delivery over the acoustic channel, the PHY layer adds stop-and-wait ARQ with automatic fragmentation:

```rust
use ggwave_voice::phy::{Phy, PhyConfig, PhyEvent};

let mut phy = Phy::new(PhyConfig::default());

// Send a message (fragments automatically if > 139 bytes)
phy.send(b"hello from the other side").unwrap();

// Main loop: feed mic input, pull speaker output, check events
let mut mic_buf = [0.0f32; 1024];
let mut spk_buf = [0.0f32; 1024];
loop {
    // read_mic(&mut mic_buf);
    phy.ingest(&mic_buf);
    phy.emit(&mut spk_buf);
    // write_speaker(&spk_buf);

    while let Some(event) = phy.poll() {
        match event {
            PhyEvent::Received(msg) => println!("got: {:?}", msg),
            PhyEvent::SendComplete => println!("ACKed"),
            PhyEvent::SendFailed => println!("gave up"),
        }
    }
    # break; // (example only)
}
```

## Architecture

| Module | Purpose |
|---|---|
| `src/lib.rs` | Public API (`encode`, `Decoder`), re-exports |
| `src/protocol.rs` | Constants (sample rate, symbol timing, FFT size, formant bands) |
| `src/formant.rs` | 2-channel symbol system: vowel alphabet, synthesis, detection, 4-bit packing |
| `src/encoder.rs` | Payload → RS-encode → symbols → synthesized audio |
| `src/decoder.rs` | Streaming state machine (Listening → Receiving → Analyzing) |
| `src/fft.rs` | FFT wrapper (Hann-windowed and raw power spectrum) |
| `src/reed_solomon.rs` | RS codec over GF(2^8) with erasure support |
| `src/rs4.rs` | RS codec over GF(2^4) for streaming FEC (symbols = field elements) |
| `src/stream.rs` | Streaming FEC codec: continuous encode/decode with block interleaving |
| `src/phy.rs` | PHY layer: stop-and-wait ARQ, fragmentation, ACK/NAK, timeout/retransmit |

## Protocol

| Parameter | Value |
|---|---|
| Sample rate | 48,000 Hz |
| F0 (fundamental) | 210 Hz (low) / 270 Hz (high) |
| Harmonics | 16 |
| Symbol duration | 50 ms + 10 ms guard |
| Symbols | 16 (8 vowels × 2 pitches = 4 bits each) |
| Preamble | 4 symbols (start + end) |
| Max payload (single message) | 140 bytes |
| Error correction | Reed-Solomon GF(2^8), adaptive ECC |

### Streaming FEC

| Parameter | Default |
|---|---|
| Codeword | RS(15,11) over GF(2^4) — 4 parity symbols |
| Error correction | 2 symbol errors or 4 erasures per codeword |
| Interleaving | Block interleaver, depth 2 |
| Throughput | ~6 B/s (3–4× over ARQ) |
| Configurable | `rs_parity` (2–8), `interleave_depth` (1–4) |

The 4-bit modem symbols are GF(2^4) field elements — RS operates directly on modem symbols with no byte packing between FEC and modulation.

### PHY (reliable delivery)

| Parameter | Default |
|---|---|
| Max message size | 2,224 bytes (16 fragments × 139 bytes) |
| ACK timeout | 4,000 ms |
| Max retries | 5 per fragment |
| Volume | 50 (0–100) |

The PHY is a pure state machine — no threads, no timers, no I/O. WASM-compatible.

## Robustness

163 tests across unit, integration, and channel simulation:

- Phone line (300–3400 Hz bandpass + 8 kHz resample + μ-law codec + noise)
- AM radio (300–2500 Hz bandpass + echo + noise)
- VoIP narrowband (400–3000 Hz + 8-bit quantization)
- Neural noise cancellation (RNNoise — Mozilla/Xiph noise suppressor)
- Hard/soft clipping and saturation
- Frequency drift (±5 Hz Doppler)
- Room reverb (multiple echo paths)
- Wow/flutter (speed wobble)
- Notch filters (room modes / interference)
- Signal fading (time-varying gain)
- Low-resolution ADC (down to 6-bit)
- Combined nightmare channels (all of the above stacked)

SNR threshold: ~10 dB through a phone channel. Survives RNNoise at ≥20 dB SNR.

All 256 single-byte values and all payload lengths 1–140 round-trip correctly.

## Dependencies

- [`rustfft`](https://crates.io/crates/rustfft) — FFT computation
- [`thiserror`](https://crates.io/crates/thiserror) — error types
- [`hound`](https://crates.io/crates/hound) — WAV I/O (dev only)
- [`criterion`](https://crates.io/crates/criterion) — benchmarking (dev only)
- [`nnnoiseless`](https://crates.io/crates/nnnoiseless) — RNNoise noise suppression (dev only)

## License

MIT
