# sóng
*sống* 🌊

Data as song. Survives VoIP and noise cancellation.

## Teams/VoIP Branch

This branch is optimized for **Microsoft Teams, Zoom, and VoIP codecs** which aggressively filter the F1 formant band (400-850 Hz). It uses F2-only detection with hybrid F1+F2 fallback for clean channels.

**Trade-off:** Lower throughput (~30 bps vs ~40 bps) for better VoIP survival.

For the full 8-vowel version, see the `master` branch.

## The Problem

Acoustic modems like ggwave use pure sine tones. These get destroyed by phone call codecs (AMR, Opus), noise cancellation (Krisp, WebRTC), and speaker-air-microphone chains.

VoIP platforms like Teams/Zoom are even more aggressive — they crush the F1 band entirely while boosting low frequencies.

## The Insight

Noise cancellation preserves human voice. So we encode data as voice-like signals — formant positions instead of pure tones, harmonic structure instead of sine waves. If it sounds like speech, it survives.

## How It Works

Data is encoded as a sequence of **vowel-like sounds** using two orthogonal channels:

- **Vowel channel** (2 bits): F2 formant frequency selects one of 4 vowels (F2-only for VoIP robustness)
- **Pitch channel** (1 bit): F0 fundamental frequency — 208 Hz G#3 (low) or 277 Hz C#4 (high)

Each symbol carries 3 bits (8 symbols total), synthesized as a harmonic series shaped by formant envelopes.

**Hybrid detection:** Runs F2-only and F1+F2 classifiers in parallel, picking the higher-confidence result. This gives VoIP robustness while maintaining accuracy on clean channels.

4 VoIP-optimized vowels (F2 positions survive Teams/Zoom processing):

| Vowel | F1 (Hz) | F2 (Hz) | Note |
|-------|---------|---------|------|
| 0 | 416 | 1074 | low F1, low F2 |
| 1 | 624 | 1420 | mid F1, mid-low F2 |
| 2 | 520 | 1663 | mid F1, mid-high F2 |
| 3 | 832 | 2252 | high F1, high F2 |

F1 values are used by the hybrid detector when available (clean channels), but F2-only detection handles VoIP where F1 is crushed.

## Speed Comparison

| Method | Throughput |
|--------|------------|
| Morse code (20 WPM) | ~10 bps |
| Morse code (expert, 40 WPM) | ~20 bps |
| **This modem (Teams branch)** | **~30 bps** |
| Main branch (8 vowels) | ~40 bps |

About 2-3x faster than a skilled WW1 morse operator, and it survives modern VoIP channels that would destroy traditional morse (VAD, noise suppression, AGC).

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

### Play a message

```bash
cargo run --example play --release -- "What hath God wrought?"
```

### PHY layer (reliable delivery)

For reliable delivery over the acoustic channel with automatic fragmentation and retransmit:

```rust
use ggwave_voice::phy::{Phy, PhyConfig, PhyEvent};

let mut phy = Phy::new(PhyConfig::default());
phy.send(b"hello from the other side").unwrap();

loop {
    phy.ingest(&mic_buf);
    phy.emit(&mut spk_buf);

    while let Some(event) = phy.poll() {
        match event {
            PhyEvent::Received(msg) => println!("got: {:?}", msg),
            PhyEvent::SendComplete => println!("ACKed"),
            PhyEvent::SendFailed => println!("gave up"),
        }
    }
}
```

## Protocol

| Parameter | Value |
|---|---|
| Sample rate | 48,000 Hz |
| F0 (fundamental) | 208 Hz G#3 / 277 Hz C#4 |
| Harmonics | 16 |
| Symbol duration | 50 ms + 10 ms guard |
| Symbols | 8 (4 vowels × 2 pitches = 3 bits each) |
| Preamble | 4 symbols (start + end) |
| Max payload | 140 bytes |
| Error correction | Reed-Solomon GF(2^8), adaptive ECC |
| Detection | Hybrid F2-only + F1+F2 with confidence voting |

## Robustness

All 138 tests pass including channel simulations:

- Phone line (300–3400 Hz bandpass + 8 kHz resample + μ-law codec + noise)
- VoIP narrowband (400–3000 Hz + quantization)
- Neural noise cancellation (RNNoise)
- Room reverb and speakerphone scenarios
- Hard/soft clipping
- Frequency drift (±5 Hz)
- Signal fading
- Combined nightmare channels

SNR threshold: ~10 dB through a phone channel. Survives RNNoise at ≥20 dB SNR.

## License

MIT
