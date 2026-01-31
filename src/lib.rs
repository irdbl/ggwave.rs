//! Rust port of ggwave's AUDIBLE_FAST codec.
//!
//! Encode text to 48 kHz f32 audio and decode f32 audio back to text.
//!
//! # Example
//!
//! ```
//! let audio = ggwave::encode(b"hello", 25).unwrap();
//! let mut decoder = ggwave::Decoder::new();
//! let payload = decoder.decode(&audio).unwrap().unwrap();
//! assert_eq!(&payload, b"hello");
//! ```

pub mod protocol;
pub mod reed_solomon;
pub mod dss;
pub mod fft;
pub mod encoder;
pub mod decoder;

pub use decoder::Decoder;

/// Encode a payload into f32 audio samples using the AUDIBLE_FAST protocol.
///
/// `payload`: the bytes to transmit (max 140 bytes).
/// `volume`: output volume, 0..=100 (typically 10-50).
///
/// Returns a `Vec<f32>` of audio samples at 48 kHz.
pub fn encode(payload: &[u8], volume: u8) -> Result<Vec<f32>, Error> {
    encoder::encode(payload, volume)
}

/// Errors returned by encode/decode operations.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("payload is empty")]
    EmptyPayload,

    #[error("payload too large: {size} bytes (max {max})")]
    PayloadTooLarge { size: usize, max: usize },

    #[error("invalid volume: {0} (must be 0..=100)")]
    InvalidVolume(u8),

    #[error("decode failed: could not extract valid payload from audio")]
    DecodeFailed,
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- Parameter validation (ported from test-ggwave.cpp lines 180-201) ---

    #[test]
    fn test_encode_empty_payload() {
        assert!(matches!(encode(b"", 25), Err(Error::EmptyPayload)));
    }

    #[test]
    fn test_encode_too_large() {
        let big = vec![0u8; 141];
        assert!(matches!(
            encode(&big, 25),
            Err(Error::PayloadTooLarge { .. })
        ));
    }

    #[test]
    fn test_encode_volume_bounds() {
        // volume 0 should succeed
        assert!(encode(b"hello", 0).is_ok());
        // volume 100 should succeed
        assert!(encode(b"hello", 100).is_ok());
        // volume 101 should fail
        assert!(matches!(
            encode(b"hello", 101),
            Err(Error::InvalidVolume(101))
        ));
        // volume 50 should succeed
        assert!(encode(b"hello", 50).is_ok());
    }

    #[test]
    fn test_encode_various_sizes() {
        // Valid sizes 1..=3 with small payload (mirrors C++ init checks)
        for size in 1..=3 {
            let payload = &b"asd"[..size];
            assert!(encode(payload, 25).is_ok(), "encode failed for size {size}");
        }
    }

    // --- Encode size prediction (ported from test-ggwave.cpp lines 275-278) ---

    #[test]
    fn test_encode_size_prediction() {
        // The number of output samples should match the formula:
        // (N_MARKER_FRAMES + total_data_frames + N_MARKER_FRAMES) * SAMPLES_PER_FRAME
        let payload = b"a0Z5kR2g";
        let audio = encode(payload, 25).unwrap();

        let data_length = payload.len();
        let n_ecc = protocol::ecc_bytes_for_length(data_length);
        let total_bytes = protocol::ENCODED_DATA_OFFSET + data_length + n_ecc;
        let total_data_frames = protocol::EXTRA
            * ((total_bytes + protocol::BYTES_PER_TX - 1) / protocol::BYTES_PER_TX)
            * protocol::FRAMES_PER_TX;
        let expected_samples = (protocol::N_MARKER_FRAMES + total_data_frames + protocol::N_MARKER_FRAMES)
            * protocol::SAMPLES_PER_FRAME;

        assert_eq!(
            audio.len(),
            expected_samples,
            "encode size mismatch: got {} expected {}",
            audio.len(),
            expected_samples
        );
    }

    // --- Variable-length round-trip for every length 1..8 ---
    // (ported from test-ggwave.cpp lines 257-289)

    #[test]
    fn test_roundtrip_incremental_lengths() {
        let payload_full = b"a0Z5kR2g";
        for length in 1..=payload_full.len() {
            let payload = &payload_full[..length];
            let audio = encode(payload, 25).unwrap();
            let mut decoder = Decoder::new();
            let result = decoder.decode(&audio).unwrap();
            assert_eq!(
                result.as_deref(),
                Some(payload),
                "round-trip failed for length {length}"
            );
        }
    }

    // --- Noise robustness (ported from test-ggwave.cpp lines 280-281) ---

    #[test]
    fn test_roundtrip_with_noise() {
        let payload = b"a0Z5kR2g";
        let mut audio = encode(payload, 25).unwrap();

        // Add noise at 0.02 level (matching C++ variable-length test)
        let mut rng_state: u32 = 12345;
        for sample in audio.iter_mut() {
            // Simple LCG PRNG for deterministic noise
            rng_state = rng_state.wrapping_mul(1103515245).wrapping_add(12345);
            let frand = (rng_state >> 16) as f32 / 65535.0;
            *sample += (frand - 0.5) * 0.02;
            *sample = sample.clamp(-1.0, 1.0);
        }

        let mut decoder = Decoder::new();
        let result = decoder.decode(&audio).unwrap();
        assert_eq!(result.as_deref(), Some(&payload[..]));
    }

    // --- Single byte (edge case: smallest payload) ---

    #[test]
    fn test_roundtrip_single_byte() {
        let payload = b"A";
        let audio = encode(payload, 50).unwrap();
        let mut decoder = Decoder::new();
        let result = decoder.decode(&audio).unwrap();
        assert_eq!(result.as_deref(), Some(&payload[..]));
    }

    // --- "hello" basic round-trip ---

    #[test]
    fn test_roundtrip_hello() {
        let payload = b"hello";
        let audio = encode(payload, 50).unwrap();
        let mut decoder = Decoder::new();
        let result = decoder.decode(&audio).unwrap();
        assert_eq!(result.as_deref(), Some(&payload[..]));
    }

    // --- Medium payload ---

    #[test]
    fn test_roundtrip_medium() {
        let payload = b"The quick brown fox jumps over the lazy dog!12345";
        let audio = encode(payload, 50).unwrap();
        let mut decoder = Decoder::new();
        let result = decoder.decode(&audio).unwrap();
        assert_eq!(result.as_deref(), Some(&payload[..]));
    }

    // --- Max length (140 bytes) ---

    #[test]
    fn test_roundtrip_max_length() {
        let payload: Vec<u8> = (0..140).map(|i| (i % 256) as u8).collect();
        let audio = encode(&payload, 50).unwrap();
        let mut decoder = Decoder::new();
        let result = decoder.decode(&audio).unwrap();
        assert_eq!(result.as_deref(), Some(&payload[..]));
    }

    // --- Binary payload (non-ASCII, all byte values) ---

    #[test]
    fn test_roundtrip_binary_payload() {
        // Payload with bytes 0x00..0x1F — exercises low nibble values
        let payload: Vec<u8> = (0..32).collect();
        let audio = encode(&payload, 50).unwrap();
        let mut decoder = Decoder::new();
        let result = decoder.decode(&audio).unwrap();
        assert_eq!(result.as_deref(), Some(&payload[..]));
    }

    // --- Streaming decode (feed in sub-frame chunks) ---

    #[test]
    fn test_roundtrip_streaming() {
        let payload = b"stream test";
        let audio = encode(payload, 50).unwrap();
        let mut decoder = Decoder::new();

        let chunk_size = 512; // half a frame
        let mut result = None;
        for chunk in audio.chunks(chunk_size) {
            if let Ok(Some(data)) = decoder.decode(chunk) {
                result = Some(data);
                break;
            }
        }
        assert_eq!(result.as_deref(), Some(&payload[..]));
    }

    // --- Streaming with very small chunks ---

    #[test]
    fn test_roundtrip_streaming_tiny_chunks() {
        let payload = b"tiny";
        let audio = encode(payload, 50).unwrap();
        let mut decoder = Decoder::new();

        let chunk_size = 100; // much less than a frame
        let mut result = None;
        for chunk in audio.chunks(chunk_size) {
            if let Ok(Some(data)) = decoder.decode(chunk) {
                result = Some(data);
                break;
            }
        }
        assert_eq!(result.as_deref(), Some(&payload[..]));
    }

    // --- Decoder reset ---

    #[test]
    fn test_decoder_reset() {
        let mut decoder = Decoder::new();
        let payload = b"test";
        let audio = encode(payload, 50).unwrap();
        let _ = decoder.decode(&audio);
        decoder.reset();
        // After reset, should be able to decode again
        let result = decoder.decode(&audio).unwrap();
        assert_eq!(result.as_deref(), Some(&payload[..]));
    }

    // --- Volume 0 produces silent but decodable output ---

    #[test]
    fn test_encode_volume_zero_produces_silence() {
        let audio = encode(b"hi", 0).unwrap();
        // All samples should be exactly 0.0
        assert!(audio.iter().all(|&s| s == 0.0));
    }

    // --- ECC bytes calculation matches C++ ---

    #[test]
    fn test_ecc_bytes_for_length() {
        // len < 4 => 2
        assert_eq!(protocol::ecc_bytes_for_length(1), 2);
        assert_eq!(protocol::ecc_bytes_for_length(2), 2);
        assert_eq!(protocol::ecc_bytes_for_length(3), 2);
        // len = 4 => max(4, 2*(4/5)) = max(4, 0) = 4
        assert_eq!(protocol::ecc_bytes_for_length(4), 4);
        // len = 5 => max(4, 2*(5/5)) = max(4, 2) = 4
        assert_eq!(protocol::ecc_bytes_for_length(5), 4);
        // len = 10 => max(4, 2*(10/5)) = max(4, 4) = 4
        assert_eq!(protocol::ecc_bytes_for_length(10), 4);
        // len = 15 => max(4, 2*(15/5)) = max(4, 6) = 6
        assert_eq!(protocol::ecc_bytes_for_length(15), 6);
        // len = 50 => max(4, 2*(50/5)) = max(4, 20) = 20
        assert_eq!(protocol::ecc_bytes_for_length(50), 20);
        // len = 140 => max(4, 2*(140/5)) = max(4, 56) = 56
        assert_eq!(protocol::ecc_bytes_for_length(140), 56);
    }

    // --- Noise robustness at higher noise level ---

    #[test]
    fn test_roundtrip_with_moderate_noise() {
        // Test with each length 1..4 at slightly higher noise to stress RS correction
        let payload_full = b"a0Z5";
        for length in 1..=payload_full.len() {
            let payload = &payload_full[..length];
            let mut audio = encode(payload, 50).unwrap();

            let mut rng_state: u32 = 42 + length as u32;
            for sample in audio.iter_mut() {
                rng_state = rng_state.wrapping_mul(1103515245).wrapping_add(12345);
                let frand = (rng_state >> 16) as f32 / 65535.0;
                *sample += (frand - 0.5) * 0.01;
                *sample = sample.clamp(-1.0, 1.0);
            }

            let mut decoder = Decoder::new();
            let result = decoder.decode(&audio).unwrap();
            assert_eq!(
                result.as_deref(),
                Some(payload),
                "noisy round-trip failed for length {length}"
            );
        }
    }

    // --- ECC boundary lengths (stress RS code word boundaries) ---

    #[test]
    fn test_roundtrip_ecc_boundaries() {
        // These lengths trigger ECC byte count transitions:
        // len=3 -> 2 ECC, len=4 -> 4 ECC, len=5 -> 4 ECC,
        // len=10 -> 4 ECC, len=15 -> 6 ECC, len=25 -> 10 ECC
        let test_lengths = [3, 4, 5, 10, 15, 25, 50, 100, 140];
        for &len in &test_lengths {
            let payload: Vec<u8> = (0..len).map(|i| ((i * 7 + 13) % 256) as u8).collect();
            let audio = encode(&payload, 50).unwrap();
            let mut decoder = Decoder::new();
            let result = decoder.decode(&audio).unwrap();
            assert_eq!(
                result.as_deref(),
                Some(&payload[..]),
                "ECC boundary round-trip failed for length {len}"
            );
        }
    }

    // --- All same byte values (degenerate payloads) ---

    #[test]
    fn test_roundtrip_all_zeros() {
        let payload = vec![0u8; 16];
        let audio = encode(&payload, 50).unwrap();
        let mut decoder = Decoder::new();
        let result = decoder.decode(&audio).unwrap();
        assert_eq!(result.as_deref(), Some(&payload[..]));
    }

    #[test]
    fn test_roundtrip_all_ones() {
        let payload = vec![0xFFu8; 16];
        let audio = encode(&payload, 50).unwrap();
        let mut decoder = Decoder::new();
        let result = decoder.decode(&audio).unwrap();
        assert_eq!(result.as_deref(), Some(&payload[..]));
    }

    // --- Decoder handles silence without panicking ---

    #[test]
    fn test_decode_silence_returns_none() {
        let silence = vec![0.0f32; 48000]; // 1 second of silence
        let mut decoder = Decoder::new();
        let result = decoder.decode(&silence).unwrap();
        assert_eq!(result, None, "silence should produce None, not a decode");
    }

    // --- Decoder handles random noise without panicking ---

    #[test]
    fn test_decode_random_noise_no_panic() {
        let mut rng_state: u32 = 99999;
        let noise: Vec<f32> = (0..48000)
            .map(|_| {
                rng_state = rng_state.wrapping_mul(1103515245).wrapping_add(12345);
                let f = (rng_state >> 16) as f32 / 65535.0;
                (f - 0.5) * 2.0
            })
            .collect();
        let mut decoder = Decoder::new();
        // Should not panic — either returns None or an error
        let _ = decoder.decode(&noise);
    }

    // --- Multiple sequential decodes on one decoder instance ---

    #[test]
    fn test_sequential_decodes() {
        let payloads: &[&[u8]] = &[b"first", b"second", b"third"];
        let mut decoder = Decoder::new();
        for payload in payloads {
            let audio = encode(payload, 50).unwrap();
            decoder.reset();
            let result = decoder.decode(&audio).unwrap();
            assert_eq!(
                result.as_deref(),
                Some(*payload),
                "sequential decode failed for {:?}",
                String::from_utf8_lossy(payload)
            );
        }
    }

    // --- Volume sweep: ensure round-trip works at various volumes ---

    #[test]
    fn test_roundtrip_volume_sweep() {
        let payload = b"volume test";
        for volume in [1, 5, 10, 25, 50, 75, 100] {
            let audio = encode(payload, volume).unwrap();
            let mut decoder = Decoder::new();
            let result = decoder.decode(&audio).unwrap();
            assert_eq!(
                result.as_deref(),
                Some(&payload[..]),
                "round-trip failed at volume {volume}"
            );
        }
    }

    // --- Encode determinism: same input → same output ---

    #[test]
    fn test_encode_deterministic() {
        let payload = b"deterministic";
        let audio1 = encode(payload, 50).unwrap();
        let audio2 = encode(payload, 50).unwrap();
        assert_eq!(audio1.len(), audio2.len());
        for (i, (a, b)) in audio1.iter().zip(audio2.iter()).enumerate() {
            assert_eq!(a, b, "sample {i} differs between two encodes");
        }
    }

    // --- Decode audio with trailing garbage after end marker ---

    #[test]
    fn test_roundtrip_with_trailing_noise() {
        let payload = b"trail";
        let mut audio = encode(payload, 50).unwrap();

        // Append 1 second of random noise after the signal
        let mut rng_state: u32 = 77777;
        let noise: Vec<f32> = (0..48000)
            .map(|_| {
                rng_state = rng_state.wrapping_mul(1103515245).wrapping_add(12345);
                let f = (rng_state >> 16) as f32 / 65535.0;
                (f - 0.5) * 0.5
            })
            .collect();
        audio.extend_from_slice(&noise);

        let mut decoder = Decoder::new();
        let result = decoder.decode(&audio).unwrap();
        assert_eq!(result.as_deref(), Some(&payload[..]));
    }

    // --- Decode with both leading silence and trailing noise ---

    #[test]
    fn test_roundtrip_padded_both_sides() {
        let payload = b"padded";
        let audio = encode(payload, 50).unwrap();

        let mut padded = vec![0.0f32; 24000]; // 0.5s leading silence
        padded.extend_from_slice(&audio);
        // 0.5s trailing silence
        padded.extend(vec![0.0f32; 24000]);

        let mut decoder = Decoder::new();
        let result = decoder.decode(&padded).unwrap();
        assert_eq!(result.as_deref(), Some(&payload[..]));
    }

    // --- Sequential messages with decoder reset ---

    #[test]
    fn test_roundtrip_4byte_payloads() {
        // Systematic test of 4-byte payloads
        let test_cases: &[&[u8]] = &[
            b"msg1", b"msg2", b"msg3",
            b"test", b"abcd", b"ABCD",
            b"0000", b"1234", b"wxyz",
        ];
        let mut failures = Vec::new();
        for &payload in test_cases {
            let audio = encode(payload, 50).unwrap();
            let mut decoder = Decoder::new();
            match decoder.decode(&audio) {
                Ok(Some(data)) if data == payload => {}
                Ok(Some(data)) => failures.push((payload, format!("wrong: {:?}", data))),
                Ok(None) => failures.push((payload, "None".into())),
                Err(e) => failures.push((payload, format!("Err: {e}"))),
            }
        }
        if !failures.is_empty() {
            for (payload, msg) in &failures {
                eprintln!("FAIL {:?}: {msg}", String::from_utf8_lossy(payload));
            }
            panic!("{} of {} 4-byte payloads failed", failures.len(), test_cases.len());
        }
    }

    // --- Spot-check previously-problematic single byte values ---

    #[test]
    fn test_roundtrip_problematic_single_bytes() {
        // These specific byte values triggered false-positive RS decodes
        // at wrong sub-frame offsets before the confidence scoring fix.
        for &b in &[0x02u8, 0x03, 0x12, 0x32, 0x42, 0x62, 0x72, 0x73, 0x6D] {
            let payload = [b];
            let audio = encode(&payload, 50).unwrap();
            let mut decoder = Decoder::new();
            let result = decoder.decode(&audio).unwrap();
            assert_eq!(
                result.as_deref(),
                Some(&payload[..]),
                "single-byte round-trip failed for byte 0x{b:02X}"
            );
        }
    }

    // --- Sample count matches for boundary payload sizes ---

    #[test]
    fn test_encode_sample_counts_consistent() {
        // Verify the output length formula holds for various sizes
        for size in [1, 2, 3, 4, 5, 10, 15, 25, 50, 100, 140] {
            let payload: Vec<u8> = (0..size).map(|i| (i % 256) as u8).collect();
            let audio = encode(&payload, 50).unwrap();

            let n_ecc = protocol::ecc_bytes_for_length(size);
            let total_bytes = protocol::ENCODED_DATA_OFFSET + size + n_ecc;
            let total_data_frames = protocol::EXTRA
                * ((total_bytes + protocol::BYTES_PER_TX - 1) / protocol::BYTES_PER_TX)
                * protocol::FRAMES_PER_TX;
            let expected = (protocol::N_MARKER_FRAMES + total_data_frames + protocol::N_MARKER_FRAMES)
                * protocol::SAMPLES_PER_FRAME;

            assert_eq!(
                audio.len(),
                expected,
                "sample count mismatch for payload size {size}: got {} expected {}",
                audio.len(),
                expected
            );
        }
    }

    // --- Every payload length 1..=140 round-trips correctly ---

    #[test]
    fn test_roundtrip_every_length() {
        let mut failures = Vec::new();
        for len in 1..=140 {
            // Deterministic payload based on length
            let payload: Vec<u8> = (0..len).map(|i| ((i * 7 + len * 3) % 256) as u8).collect();
            let audio = encode(&payload, 50).unwrap();
            let mut decoder = Decoder::new();
            match decoder.decode(&audio) {
                Ok(Some(data)) if data == payload => {}
                Ok(Some(data)) => {
                    failures.push((len, format!("wrong data (got {} bytes)", data.len())));
                }
                Ok(None) => failures.push((len, "None".into())),
                Err(e) => failures.push((len, format!("Err: {e}"))),
            }
        }
        if !failures.is_empty() {
            for (len, msg) in &failures {
                eprintln!("FAIL length {len}: {msg}");
            }
            panic!("{} of 140 lengths failed", failures.len());
        }
    }

    // --- 16-bit quantization simulation (float → i16 → float → decode) ---

    #[test]
    fn test_roundtrip_16bit_quantization() {
        let cases: &[&[u8]] = &[b"hello", b"A", b"test1234", b"\x00\xFF\x80\x7F"];
        for &payload in cases {
            let audio = encode(payload, 50).unwrap();

            // Simulate 16-bit WAV quantization
            let quantized: Vec<f32> = audio
                .iter()
                .map(|&s| {
                    let i16_val = (s * 32767.0).clamp(-32768.0, 32767.0) as i16;
                    i16_val as f32 / 32768.0
                })
                .collect();

            let mut decoder = Decoder::new();
            let result = decoder.decode(&quantized).unwrap();
            assert_eq!(
                result.as_deref(),
                Some(payload),
                "16-bit quantization round-trip failed for {:?}",
                String::from_utf8_lossy(payload)
            );
        }
    }

    // --- Truncated audio should not panic ---

    #[test]
    fn test_truncated_audio_no_panic() {
        let payload = b"truncate me";
        let audio = encode(payload, 50).unwrap();

        // Try decoding at 25%, 50%, 75% of the audio
        for &frac in &[0.25, 0.50, 0.75] {
            let end = (audio.len() as f64 * frac) as usize;
            let truncated = &audio[..end];
            let mut decoder = Decoder::new();
            // Should not panic — either None or Err is acceptable
            let _ = decoder.decode(truncated);
        }
    }

    // --- Audio with DC offset should still decode ---

    #[test]
    fn test_roundtrip_with_dc_offset() {
        let payload = b"dc offset";
        let audio = encode(payload, 50).unwrap();

        // Add a DC offset
        let offset_audio: Vec<f32> = audio.iter().map(|&s| s + 0.1).collect();

        let mut decoder = Decoder::new();
        let result = decoder.decode(&offset_audio).unwrap();
        assert_eq!(result.as_deref(), Some(&payload[..]));
    }

    // --- Repeated same-byte payloads at various lengths ---

    #[test]
    fn test_roundtrip_repeated_bytes() {
        // Single repeated byte at different lengths exercises RS with uniform data
        for &byte_val in &[0x00, 0x55, 0xAA, 0xFF] {
            for len in [1, 3, 5, 10, 20] {
                let payload = vec![byte_val; len];
                let audio = encode(&payload, 50).unwrap();
                let mut decoder = Decoder::new();
                let result = decoder.decode(&audio).unwrap();
                assert_eq!(
                    result.as_deref(),
                    Some(&payload[..]),
                    "repeated 0x{byte_val:02X} x{len} failed"
                );
            }
        }
    }

    // --- Low volume robustness ---

    #[test]
    fn test_roundtrip_low_volume() {
        // Very low volumes are harder to decode; ensure they still work
        let payload = b"quiet signal";
        for volume in [1, 2, 3, 5] {
            let audio = encode(payload, volume).unwrap();
            let mut decoder = Decoder::new();
            let result = decoder.decode(&audio).unwrap();
            assert_eq!(
                result.as_deref(),
                Some(&payload[..]),
                "low volume {volume} round-trip failed"
            );
        }
    }

    // --- Alternating nibble patterns ---

    #[test]
    fn test_roundtrip_alternating_nibbles() {
        // Payloads that alternate between extreme nibble values
        let patterns: &[&[u8]] = &[
            &[0x0F, 0xF0, 0x0F, 0xF0, 0x0F],
            &[0x55, 0xAA, 0x55, 0xAA, 0x55],
            &[0x01, 0x10, 0x01, 0x10, 0x01],
            &[0xFE, 0xEF, 0xFE, 0xEF, 0xFE],
        ];
        for &payload in patterns {
            let audio = encode(payload, 50).unwrap();
            let mut decoder = Decoder::new();
            let result = decoder.decode(&audio).unwrap();
            assert_eq!(
                result.as_deref(),
                Some(payload),
                "alternating pattern {payload:02X?} failed"
            );
        }
    }

    // --- Back-to-back messages without explicit reset ---

    #[test]
    fn test_back_to_back_no_reset() {
        // After a successful decode, the decoder should transition back to
        // Listening state. Feeding another message should work without reset.
        let payloads: &[&[u8]] = &[b"first", b"second"];
        let mut decoder = Decoder::new();
        for &payload in payloads {
            let audio = encode(payload, 50).unwrap();
            // Add silence gap between messages
            let mut padded = vec![0.0f32; 24000];
            padded.extend_from_slice(&audio);
            padded.extend(vec![0.0f32; 24000]);
            let result = decoder.decode(&padded).unwrap();
            assert_eq!(
                result.as_deref(),
                Some(payload),
                "back-to-back (no reset) failed for {:?}",
                String::from_utf8_lossy(payload)
            );
        }
    }

    // --- Degenerate payloads at ECC transition boundaries ---

    #[test]
    fn test_roundtrip_degenerate_at_ecc_boundaries() {
        // All-zeros and all-FF payloads at lengths where ECC byte count changes.
        // These are the hardest cases: uniform data forms valid RS codewords
        // at shorter lengths, testing the confidence scoring discrimination.
        let ecc_boundary_lengths = [3, 4, 5, 10, 15, 20, 25, 50];
        for &len in &ecc_boundary_lengths {
            for &byte_val in &[0x00, 0xFF] {
                let payload = vec![byte_val; len];
                let audio = encode(&payload, 50).unwrap();
                let mut decoder = Decoder::new();
                let result = decoder.decode(&audio).unwrap();
                assert_eq!(
                    result.as_deref(),
                    Some(&payload[..]),
                    "degenerate 0x{byte_val:02X} x{len} at ECC boundary failed"
                );
            }
        }
    }

    // --- Sampled 2-byte round-trip (covers nibble pair interactions) ---

    #[test]
    fn test_roundtrip_2byte_sampled() {
        // Test all 16 * 16 = 256 combinations of low nibbles (high nibbles = 0)
        // plus all combinations of high nibbles (low nibbles = 0).
        // This catches nibble-pair interaction bugs without testing all 65536 pairs.
        let mut failures = Vec::new();

        // Low nibble sweep: [0x0N, 0x0M] for N,M in 0..16
        for lo1 in 0..16u8 {
            for lo2 in 0..16u8 {
                let payload = [lo1, lo2];
                let audio = encode(&payload, 50).unwrap();
                let mut decoder = Decoder::new();
                match decoder.decode(&audio) {
                    Ok(Some(data)) if data == payload => {}
                    other => failures.push((payload, format!("{other:?}"))),
                }
            }
        }

        // High nibble sweep: [0xN0, 0xM0] for N,M in 0..16
        for hi1 in 0..16u8 {
            for hi2 in 0..16u8 {
                let payload = [hi1 << 4, hi2 << 4];
                let audio = encode(&payload, 50).unwrap();
                let mut decoder = Decoder::new();
                match decoder.decode(&audio) {
                    Ok(Some(data)) if data == payload => {}
                    other => failures.push((payload, format!("{other:?}"))),
                }
            }
        }

        if !failures.is_empty() {
            for (payload, msg) in &failures[..failures.len().min(10)] {
                eprintln!("FAIL [{:02X}, {:02X}]: {msg}", payload[0], payload[1]);
            }
            panic!("{} of 512 2-byte combinations failed", failures.len());
        }
    }

    // --- Noise sweep: find the noise tolerance threshold ---

    #[test]
    fn test_noise_tolerance_sweep() {
        let payload = b"noise test";
        let audio = encode(payload, 50).unwrap();

        // Test increasing noise levels until decode fails
        let mut max_tolerated = 0.0f64;
        for noise_pct in [0.01, 0.02, 0.05, 0.10, 0.15, 0.20] {
            let mut noisy = audio.clone();
            let mut rng_state: u32 = (noise_pct * 100000.0) as u32;
            for sample in noisy.iter_mut() {
                rng_state = rng_state.wrapping_mul(1103515245).wrapping_add(12345);
                let frand = (rng_state >> 16) as f32 / 65535.0;
                *sample += (frand - 0.5) * noise_pct as f32;
                *sample = sample.clamp(-1.0, 1.0);
            }

            let mut decoder = Decoder::new();
            if let Ok(Some(data)) = decoder.decode(&noisy) {
                if data == payload {
                    max_tolerated = noise_pct;
                }
            }
        }

        // Should tolerate at least 5% noise at volume 50
        assert!(
            max_tolerated >= 0.05,
            "noise tolerance too low: only tolerates {:.0}%",
            max_tolerated * 100.0
        );
    }

    // --- Amplitude clipping robustness ---

    #[test]
    fn test_roundtrip_clipped_audio() {
        let payload = b"clipped";
        let audio = encode(payload, 100).unwrap();

        // Hard clip to [-0.5, 0.5] (simulates ADC clipping)
        let clipped: Vec<f32> = audio.iter().map(|&s| s.clamp(-0.5, 0.5)).collect();

        let mut decoder = Decoder::new();
        let result = decoder.decode(&clipped).unwrap();
        assert_eq!(result.as_deref(), Some(&payload[..]));
    }

    // --- Amplitude scaling robustness ---

    #[test]
    fn test_roundtrip_scaled_amplitude() {
        let payload = b"scale test";
        let audio = encode(payload, 50).unwrap();

        // Scale down to 10% of original amplitude
        let scaled: Vec<f32> = audio.iter().map(|&s| s * 0.1).collect();

        let mut decoder = Decoder::new();
        let result = decoder.decode(&scaled).unwrap();
        assert_eq!(result.as_deref(), Some(&payload[..]));
    }
}
