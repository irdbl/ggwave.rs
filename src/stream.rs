//! Streaming FEC codec: continuous RS-coded symbol stream with block interleaving.
//!
//! Eliminates per-frame preamble overhead and ACK round-trips. Data flows as a continuous
//! RS(7,k) coded symbol stream where each GF(2^3) symbol maps directly to a modem symbol.
//!
//! Teams branch uses 8 modem symbols (4 vowels × 2 pitches = 3 bits), which maps 1:1
//! to GF(2^3) elements, allowing efficient RS coding without symbol mapping overhead.
//!
//! ```text
//! Send: bytes -> 3-bit symbols -> RS(7,k) encode -> interleave -> synthesize -> audio
//! Recv: audio -> preamble detect -> classify -> deinterleave -> RS(7,k) decode -> bytes
//! ```

use crate::fft;
use crate::formant;
use crate::protocol::*;
use crate::rs3::ReedSolomon3;

/// Codeword length for GF(2^3): n = 2^3 - 1 = 7.
const CODEWORD_LEN: usize = 7;

/// Stream start preamble: uses max-distance vowel pair (vowels 0 and 3).
const STREAM_PREAMBLE_START: [u8; PREAMBLE_LEN] = [0, 6, 0, 6];

/// Stream end preamble: inverted pattern.
const STREAM_PREAMBLE_END: [u8; PREAMBLE_LEN] = [6, 0, 6, 0];

/// Vowel-only patterns for preamble detection (more robust than full symbol match).
const STREAM_PREAMBLE_START_VOWELS: [usize; PREAMBLE_LEN] = [0, 3, 0, 3];
/// Confidence threshold for erasure marking.
const ERASURE_THRESHOLD: f64 = 1.0;

/// Configuration for the streaming codec.
#[derive(Debug, Clone)]
pub struct StreamConfig {
    /// Parity symbols per RS codeword (default: 2). Data symbols = 7 - parity.
    /// Max parity is 6 (leaving 1 data symbol).
    pub rs_parity: u8,
    /// Block interleaver depth (default: 2). 1 = no interleaving.
    pub interleave_depth: u8,
    /// Audio volume 0..=100 (default: 50).
    pub volume: u8,
}

impl Default for StreamConfig {
    fn default() -> Self {
        Self {
            rs_parity: 2,
            interleave_depth: 2,
            volume: 50,
        }
    }
}

impl StreamConfig {
    fn data_per_codeword(&self) -> usize {
        CODEWORD_LEN - self.rs_parity as usize
    }
}

// --- Block Interleaver ---
//
// Writes D codewords row-by-row into a D×N matrix, reads column-by-column.
// A burst of B consecutive bad symbols gets spread across ceil(B/D) codewords,
// each seeing at most ceil(B/D) errors. Latency: D * N * symbol_time.

/// Block interleaver: buffers D codewords, emits them column-interleaved.
struct Interleaver {
    depth: usize,
    /// Accumulates D * CODEWORD_LEN symbols before emitting.
    buf: Vec<u8>,
    /// Interleaved output queue.
    out: Vec<u8>,
}

impl Interleaver {
    fn new(depth: usize) -> Self {
        Self {
            depth,
            buf: Vec::new(),
            out: Vec::new(),
        }
    }

    /// Push a symbol. If a full block (D codewords) is accumulated, interleave it.
    fn push(&mut self, sym: u8) {
        self.buf.push(sym);
        let block_size = self.depth * CODEWORD_LEN;
        if self.buf.len() == block_size {
            // Read column-by-column from D×N matrix (rows = codewords, cols = symbol positions)
            for col in 0..CODEWORD_LEN {
                for row in 0..self.depth {
                    self.out.push(self.buf[row * CODEWORD_LEN + col]);
                }
            }
            self.buf.clear();
        }
    }

    /// Flush any partial block by zero-padding to a full block, then interleaving.
    fn flush(&mut self) {
        if self.buf.is_empty() {
            return;
        }
        let block_size = self.depth * CODEWORD_LEN;
        while self.buf.len() < block_size {
            self.buf.push(0);
        }
        for col in 0..CODEWORD_LEN {
            for row in 0..self.depth {
                self.out.push(self.buf[row * CODEWORD_LEN + col]);
            }
        }
        self.buf.clear();
    }

    /// Pop the next interleaved symbol, if available.
    fn pop(&mut self) -> Option<u8> {
        if self.out.is_empty() {
            None
        } else {
            Some(self.out.remove(0))
        }
    }

}

/// Block deinterleaver: inverse of interleaver. Accumulates D*N symbols,
/// then reads them back in row order (undoing the column interleaving).
struct Deinterleaver {
    depth: usize,
    buf: Vec<(u8, f64)>,
    out: Vec<(u8, f64)>,
}

impl Deinterleaver {
    fn new(depth: usize) -> Self {
        Self {
            depth,
            buf: Vec::new(),
            out: Vec::new(),
        }
    }

    /// Push a (symbol, confidence) pair. If a full block is accumulated, deinterleave it.
    fn push(&mut self, sym: u8, conf: f64) {
        self.buf.push((sym, conf));
        let block_size = self.depth * CODEWORD_LEN;
        if self.buf.len() == block_size {
            // Write was column-by-column (D symbols per column, N columns).
            // To undo: read row-by-row from N×D layout (N cols, D per col).
            // Input order: col0_row0, col0_row1, ..., col0_rowD-1, col1_row0, ...
            // Output order: row0_col0, row0_col1, ..., row0_colN-1, row1_col0, ...
            for row in 0..self.depth {
                for col in 0..CODEWORD_LEN {
                    self.out.push(self.buf[col * self.depth + row]);
                }
            }
            self.buf.clear();
        }
    }

    fn pop(&mut self) -> Option<(u8, f64)> {
        if self.out.is_empty() {
            None
        } else {
            Some(self.out.remove(0))
        }
    }

    /// Flush partial block by padding and deinterleaving.
    fn flush(&mut self) {
        if self.buf.is_empty() {
            return;
        }
        let block_size = self.depth * CODEWORD_LEN;
        while self.buf.len() < block_size {
            self.buf.push((0, 0.0));
        }
        for row in 0..self.depth {
            for col in 0..CODEWORD_LEN {
                self.out.push(self.buf[col * self.depth + row]);
            }
        }
        self.buf.clear();
    }
}

// --- StreamTx (encoder) ---

/// Streaming transmitter: feeds bytes, emits continuous audio.
///
/// Usage: call `feed()` with payload bytes (length prefix is added automatically),
/// call `finish()` when done, then call `emit()` repeatedly to get audio samples.
pub struct StreamTx {
    config: StreamConfig,
    rs: ReedSolomon3,
    interleaver: Interleaver,
    /// Buffer for 3-bit symbols extracted from input bytes.
    sym3_buf: Vec<u8>,
    /// Accumulated bits from incomplete byte-to-symbol conversion.
    bit_accum: u32,
    bit_count: usize,
    symbol_queue: Vec<u8>,
    sample_pos: usize,
    current_symbol_audio: Vec<f64>,
    in_guard: bool,
    guard_pos: usize,
    preamble_sent: bool,
    flushing: bool,
    end_queued: bool,
    done: bool,
    send_volume: f64,
}

impl StreamTx {
    pub fn new(config: StreamConfig) -> Self {
        let data_len = config.data_per_codeword();
        let rs = ReedSolomon3::new(data_len, config.rs_parity as usize);
        let interleaver = Interleaver::new(config.interleave_depth as usize);
        let send_volume = config.volume as f64 / 100.0;
        Self {
            config,
            rs,
            interleaver,
            sym3_buf: Vec::new(),
            bit_accum: 0,
            bit_count: 0,
            symbol_queue: Vec::new(),
            sample_pos: 0,
            current_symbol_audio: Vec::new(),
            in_guard: false,
            guard_pos: 0,
            preamble_sent: false,
            flushing: false,
            end_queued: false,
            done: false,
            send_volume,
        }
    }

    /// Queue raw bytes for transmission (caller handles framing).
    pub fn feed_raw(&mut self, bytes: &[u8]) {
        // Convert bytes to 3-bit symbols
        for &b in bytes {
            self.bit_accum = (self.bit_accum << 8) | (b as u32);
            self.bit_count += 8;
            while self.bit_count >= 3 {
                self.bit_count -= 3;
                let sym = ((self.bit_accum >> self.bit_count) & 0x7) as u8;
                self.sym3_buf.push(sym);
            }
        }
    }

    /// Queue data bytes with an automatic length prefix.
    /// The length byte is prepended so the receiver knows how many payload bytes to expect.
    pub fn feed(&mut self, payload: &[u8]) {
        assert!(payload.len() <= 255, "stream payload max 255 bytes");
        // Length prefix: 1 byte
        let len = payload.len() as u8;
        self.feed_raw(&[len]);
        // Payload
        self.feed_raw(payload);
    }

    /// Signal that no more data will be fed. Flushes remaining nibbles with zero-padding.
    pub fn finish(&mut self) {
        self.flushing = true;
    }

    /// Fill the output buffer with audio samples. Returns number of samples written.
    pub fn emit(&mut self, out: &mut [f32]) -> usize {
        if self.done {
            return 0;
        }

        let mut written = 0;

        while written < out.len() {
            // Ensure start preamble
            if !self.preamble_sent {
                for &sym in &STREAM_PREAMBLE_START {
                    self.symbol_queue.push(sym);
                }
                self.preamble_sent = true;
            }

            // Encode nibbles into RS codewords
            self.encode_codewords();

            // Handle flushing
            if self.flushing && !self.end_queued {
                self.flush_final();
            }

            // If we have no symbols and we're done, stop
            if self.symbol_queue.is_empty()
                && self.current_symbol_audio.is_empty()
                && !self.in_guard
            {
                if self.end_queued {
                    self.done = true;
                }
                break;
            }

            // Synthesize from symbol queue
            written += self.synthesize_into(&mut out[written..]);
        }

        written
    }

    /// Returns true when all queued data has been fully emitted as audio.
    pub fn is_idle(&self) -> bool {
        self.done
    }

    fn encode_codewords(&mut self) {
        let k = self.config.data_per_codeword();
        while self.sym3_buf.len() >= k {
            let data: Vec<u8> = self.sym3_buf.drain(..k).collect();
            let encoded = self.rs.encode(&data);
            for &sym in &encoded {
                self.interleaver.push(sym);
            }
        }
        // Drain interleaver output into symbol queue
        while let Some(sym) = self.interleaver.pop() {
            self.symbol_queue.push(sym);
        }
    }

    fn flush_final(&mut self) {
        // Flush any remaining bits as a partial symbol
        if self.bit_count > 0 {
            let sym = ((self.bit_accum << (3 - self.bit_count)) & 0x7) as u8;
            self.sym3_buf.push(sym);
            self.bit_count = 0;
            self.bit_accum = 0;
        }

        let k = self.config.data_per_codeword();

        // Pad remaining symbols to fill a codeword
        if !self.sym3_buf.is_empty() {
            while self.sym3_buf.len() < k {
                self.sym3_buf.push(0);
            }
            let data: Vec<u8> = self.sym3_buf.drain(..k).collect();
            let encoded = self.rs.encode(&data);
            for &sym in &encoded {
                self.interleaver.push(sym);
            }
        }

        // Flush interleaver (pads partial block)
        self.interleaver.flush();
        while let Some(sym) = self.interleaver.pop() {
            self.symbol_queue.push(sym);
        }

        // End preamble
        for &sym in &STREAM_PREAMBLE_END {
            self.symbol_queue.push(sym);
        }

        self.end_queued = true;
    }

    fn synthesize_into(&mut self, out: &mut [f32]) -> usize {
        let mut written = 0;

        while written < out.len() {
            if self.in_guard {
                let remaining_guard = GUARD_SAMPLES - self.guard_pos;
                let to_write = remaining_guard.min(out.len() - written);
                for i in 0..to_write {
                    out[written + i] = 0.0;
                }
                written += to_write;
                self.guard_pos += to_write;
                if self.guard_pos >= GUARD_SAMPLES {
                    self.in_guard = false;
                    self.guard_pos = 0;
                }
                continue;
            }

            if !self.current_symbol_audio.is_empty() {
                let remaining = self.current_symbol_audio.len() - self.sample_pos;
                let to_write = remaining.min(out.len() - written);
                for i in 0..to_write {
                    out[written + i] = self.current_symbol_audio[self.sample_pos + i] as f32;
                }
                written += to_write;
                self.sample_pos += to_write;
                if self.sample_pos >= self.current_symbol_audio.len() {
                    self.current_symbol_audio.clear();
                    self.sample_pos = 0;
                    self.in_guard = true;
                    self.guard_pos = 0;
                }
                continue;
            }

            if self.symbol_queue.is_empty() {
                break;
            }

            let sym = self.symbol_queue.remove(0) as usize;
            self.current_symbol_audio = formant::synthesize_symbol(sym, self.send_volume);
            self.sample_pos = 0;
        }

        written
    }
}

// --- StreamRx (decoder) ---

/// Streaming receiver: ingests audio samples, outputs decoded bytes.
pub struct StreamRx {
    config: StreamConfig,
    rs: ReedSolomon3,
    deinterleaver: Deinterleaver,
    sample_buf: Vec<f32>,
    recent_vowels: Vec<usize>,
    recent_symbols: Vec<usize>,
    state: RxState,
    codeword_buf: Vec<u8>,
    codeword_conf: Vec<f64>,
    output_buf: Vec<u8>,
    /// Decoded 3-bit symbols from RS decoding.
    decoded_symbols: Vec<u8>,
    stream_symbol_count: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RxState {
    Listening,
    Streaming,
}

impl StreamRx {
    pub fn new(config: StreamConfig) -> Self {
        let data_len = config.data_per_codeword();
        let rs = ReedSolomon3::new(data_len, config.rs_parity as usize);
        let deinterleaver = Deinterleaver::new(config.interleave_depth as usize);
        Self {
            config,
            rs,
            deinterleaver,
            sample_buf: Vec::new(),
            recent_vowels: Vec::new(),
            recent_symbols: Vec::new(),
            state: RxState::Listening,
            codeword_buf: Vec::new(),
            codeword_conf: Vec::new(),
            output_buf: Vec::new(),
            decoded_symbols: Vec::new(),
            stream_symbol_count: 0,
        }
    }

    /// Feed audio samples into the receiver.
    pub fn ingest(&mut self, samples: &[f32]) {
        self.sample_buf.extend_from_slice(samples);
        self.process_samples();
    }

    /// Read decoded bytes. Returns number of bytes written to `out`.
    pub fn read(&mut self, out: &mut [u8]) -> usize {
        let n = self.output_buf.len().min(out.len());
        out[..n].copy_from_slice(&self.output_buf[..n]);
        self.output_buf.drain(..n);
        n
    }

    /// Read all available decoded bytes.
    pub fn read_all(&mut self) -> Vec<u8> {
        std::mem::take(&mut self.output_buf)
    }

    /// Returns true if the receiver is currently in the streaming state.
    pub fn is_active(&self) -> bool {
        self.state == RxState::Streaming
    }

    fn process_samples(&mut self) {
        while self.sample_buf.len() >= SYMBOL_TOTAL_SAMPLES {
            let window: Vec<f32> = self.sample_buf[..SYMBOL_TOTAL_SAMPLES].to_vec();
            self.sample_buf.drain(..SYMBOL_TOTAL_SAMPLES);
            self.process_window(&window);
        }
    }

    fn process_window(&mut self, window: &[f32]) {
        match self.state {
            RxState::Listening => {
                let vowel = classify_window_vowel(window);
                self.recent_vowels.push(vowel);
                if self.recent_vowels.len() > PREAMBLE_LEN {
                    self.recent_vowels
                        .drain(..self.recent_vowels.len() - PREAMBLE_LEN);
                }

                if self.recent_vowels.len() == PREAMBLE_LEN
                    && self.recent_vowels[..] == STREAM_PREAMBLE_START_VOWELS[..]
                {
                    self.state = RxState::Streaming;
                    self.recent_vowels.clear();
                    self.recent_symbols.clear();
                    self.codeword_buf.clear();
                    self.codeword_conf.clear();
                    self.decoded_symbols.clear();
                    self.output_buf.clear();
                    self.stream_symbol_count = 0;
                    self.deinterleaver =
                        Deinterleaver::new(self.config.interleave_depth as usize);
                }
            }
            RxState::Streaming => {
                let voiced = &window[..SAMPLES_PER_SYMBOL.min(window.len())];
                let (sym, conf) = classify_voiced_segment(voiced);

                self.recent_symbols.push(sym);
                if self.recent_symbols.len() > PREAMBLE_LEN {
                    self.recent_symbols
                        .drain(..self.recent_symbols.len() - PREAMBLE_LEN);
                }

                // End preamble detection uses full symbol match (not just vowels).
                // Full symbol match [14, 0, 14, 0] is much more specific than
                // vowel-only match [7, 0, 7, 0] and avoids false positives from data.
                let end_detected = self.recent_symbols.len() == PREAMBLE_LEN
                    && self.recent_symbols[0] == STREAM_PREAMBLE_END[0] as usize
                    && self.recent_symbols[1] == STREAM_PREAMBLE_END[1] as usize
                    && self.recent_symbols[2] == STREAM_PREAMBLE_END[2] as usize
                    && self.recent_symbols[3] == STREAM_PREAMBLE_END[3] as usize;

                if end_detected && self.stream_symbol_count >= CODEWORD_LEN {
                    self.deinterleaver.flush();
                    self.drain_deinterleaver();
                    self.state = RxState::Listening;
                    self.recent_vowels.clear();
                    self.recent_symbols.clear();
                    self.finalize_output();
                    return;
                }

                // Feed into block deinterleaver
                self.deinterleaver.push(sym as u8, conf);
                self.stream_symbol_count += 1;

                // Drain deinterleaved symbols into codeword buffer
                self.drain_deinterleaver();
            }
        }
    }

    fn drain_deinterleaver(&mut self) {
        while let Some((sym, conf)) = self.deinterleaver.pop() {
            self.codeword_buf.push(sym);
            self.codeword_conf.push(conf);

            if self.codeword_buf.len() == CODEWORD_LEN {
                self.decode_codeword();
            }
        }
    }

    fn decode_codeword(&mut self) {
        let encoded: Vec<u8> = self.codeword_buf.drain(..).collect();
        let confs: Vec<f64> = self.codeword_conf.drain(..).collect();

        // Try hard-decision decode first, then erasure fallback
        let decoded = self.rs.decode(&encoded).or_else(|| {
            let mut erasures: Vec<usize> = confs
                .iter()
                .enumerate()
                .filter(|(_, &c)| c < ERASURE_THRESHOLD)
                .map(|(i, _)| i)
                .collect();
            let max_erasures = self.config.rs_parity as usize;
            erasures.truncate(max_erasures);
            if erasures.is_empty() {
                return None;
            }
            self.rs.decode_with_erasures(&encoded, &erasures)
        });

        if let Some(data) = decoded {
            self.decoded_symbols.extend_from_slice(&data);
        }
    }

    /// Reassemble decoded 3-bit symbols into bytes using the length prefix.
    fn finalize_output(&mut self) {
        // Convert 3-bit symbols to bytes
        // First, convert all symbols to a byte stream
        let mut bit_accum: u32 = 0;
        let mut bit_count = 0;
        let mut bytes = Vec::new();

        for &sym in &self.decoded_symbols {
            bit_accum = (bit_accum << 3) | ((sym & 0x7) as u32);
            bit_count += 3;
            while bit_count >= 8 {
                bit_count -= 8;
                bytes.push(((bit_accum >> bit_count) & 0xFF) as u8);
            }
        }

        if bytes.is_empty() {
            return;
        }

        // First byte is length prefix
        let len_byte = bytes[0] as usize;
        if len_byte == 0 || len_byte > bytes.len() - 1 {
            return;
        }

        self.output_buf.clear();
        self.output_buf.extend_from_slice(&bytes[1..1 + len_byte]);
    }
}

/// Classify a window for vowel detection (used in preamble matching).
fn classify_window_vowel(window: &[f32]) -> usize {
    let voiced = &window[..SAMPLES_PER_SYMBOL.min(window.len())];
    let rms = {
        let sum_sq: f64 = voiced.iter().map(|&s| (s as f64) * (s as f64)).sum();
        (sum_sq / voiced.len() as f64).sqrt()
    };
    if rms < 1e-6 {
        return NUM_VOWELS;
    }
    let mut spectrum = vec![0.0f32; voiced.len()];
    fft::power_spectrum_raw(voiced, &mut spectrum);
    let f0 = formant::detect_f0(&spectrum);
    let f2 = formant::detect_formants(&spectrum, f0);
    let (vowel_idx, _) = formant::classify_vowel_f2(f2);
    vowel_idx
}

/// Classify a voiced segment into a full symbol with confidence.
fn classify_voiced_segment(voiced: &[f32]) -> (usize, f64) {
    let mut spectrum = vec![0.0f32; voiced.len()];
    fft::power_spectrum(voiced, &mut spectrum);
    formant::classify_symbol(&spectrum)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_interleaver_depth1_passthrough() {
        let mut il = Interleaver::new(1);
        // Depth 1: each block is 1*7 = 7 symbols, passthrough
        let data: Vec<u8> = (0..7).collect();
        for &sym in &data {
            il.push(sym);
        }
        let mut out = Vec::new();
        while let Some(s) = il.pop() {
            out.push(s);
        }
        assert_eq!(out, data);
    }

    #[test]
    fn test_interleaver_deinterleaver_roundtrip() {
        for depth in [1, 2, 3] {
            let mut il = Interleaver::new(depth);
            let mut dil = Deinterleaver::new(depth);

            // N codewords worth of data (exactly fills interleaver blocks)
            let n_codewords = depth * 2; // 2 full blocks
            let data: Vec<u8> = (0..(n_codewords * CODEWORD_LEN))
                .map(|i| (i % 8) as u8)  // GF(2^3) symbols are 0..7
                .collect();

            // Interleave
            for &sym in &data {
                il.push(sym);
            }
            let mut interleaved = Vec::new();
            while let Some(s) = il.pop() {
                interleaved.push(s);
            }
            assert_eq!(
                interleaved.len(),
                data.len(),
                "depth={depth}: interleaved length mismatch"
            );

            // Deinterleave
            for &sym in &interleaved {
                dil.push(sym, 1.0);
            }
            let mut deinterleaved = Vec::new();
            while let Some((s, _)) = dil.pop() {
                deinterleaved.push(s);
            }
            assert_eq!(
                deinterleaved, data,
                "depth={depth}: roundtrip failed"
            );
        }
    }

    #[test]
    fn test_interleaver_burst_spread() {
        // With depth=2, a burst of errors in consecutive positions in the
        // interleaved stream should be spread across 2 codewords.
        let depth = 2;
        let mut il = Interleaver::new(depth);
        let mut dil = Deinterleaver::new(depth);

        // 2 codewords * 7 symbols = 14 symbols
        let data: Vec<u8> = (0..14).map(|i| (i % 8) as u8).collect();
        for &sym in &data {
            il.push(sym);
        }
        let mut interleaved = Vec::new();
        while let Some(s) = il.pop() {
            interleaved.push(s);
        }

        // Corrupt 2 consecutive symbols in the interleaved stream
        for i in 2..4 {
            interleaved[i] = 7; // arbitrary corruption
        }

        for &sym in &interleaved {
            dil.push(sym, 1.0);
        }
        let mut deinterleaved = Vec::new();
        while let Some((s, _)) = dil.pop() {
            deinterleaved.push(s);
        }

        // After deinterleaving, the 2 corrupted symbols should be spread
        // across 2 codewords (1 error each), not concentrated in one
        let cw1 = &deinterleaved[0..7];
        let cw2 = &deinterleaved[7..14];
        let orig_cw1 = &data[0..7];
        let orig_cw2 = &data[7..14];
        let diff1: usize = cw1.iter().zip(orig_cw1).filter(|(a, b)| a != b).count();
        let diff2: usize = cw2.iter().zip(orig_cw2).filter(|(a, b)| a != b).count();
        assert_eq!(diff1 + diff2, 2, "total errors should be 2");
        assert!(diff1 <= 1 && diff2 <= 1, "errors should be spread: cw1={diff1}, cw2={diff2}");
    }

    #[test]
    fn test_stream_config_default() {
        let cfg = StreamConfig::default();
        assert_eq!(cfg.data_per_codeword(), 5); // 7 - 2 parity = 5
        assert_eq!(cfg.rs_parity, 2);
        assert_eq!(cfg.interleave_depth, 2);
    }
}
