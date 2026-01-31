//! Reed-Solomon codec over GF(2^8) with primitive polynomial 0x11d.
//!
//! Ported from ggwave's `src/reed-solomon/` for bit-compatibility.

/// GF(2^8) exponential table (512 entries, wraps around at 255).
#[rustfmt::skip]
const GF_EXP: [u8; 512] = [
    0x1, 0x2, 0x4, 0x8, 0x10, 0x20, 0x40, 0x80, 0x1d, 0x3a, 0x74, 0xe8, 0xcd, 0x87, 0x13, 0x26, 0x4c,
    0x98, 0x2d, 0x5a, 0xb4, 0x75, 0xea, 0xc9, 0x8f, 0x3, 0x6, 0xc, 0x18, 0x30, 0x60, 0xc0, 0x9d,
    0x27, 0x4e, 0x9c, 0x25, 0x4a, 0x94, 0x35, 0x6a, 0xd4, 0xb5, 0x77, 0xee, 0xc1, 0x9f, 0x23, 0x46,
    0x8c, 0x5, 0xa, 0x14, 0x28, 0x50, 0xa0, 0x5d, 0xba, 0x69, 0xd2, 0xb9, 0x6f, 0xde, 0xa1, 0x5f,
    0xbe, 0x61, 0xc2, 0x99, 0x2f, 0x5e, 0xbc, 0x65, 0xca, 0x89, 0xf, 0x1e, 0x3c, 0x78, 0xf0, 0xfd,
    0xe7, 0xd3, 0xbb, 0x6b, 0xd6, 0xb1, 0x7f, 0xfe, 0xe1, 0xdf, 0xa3, 0x5b, 0xb6, 0x71, 0xe2, 0xd9,
    0xaf, 0x43, 0x86, 0x11, 0x22, 0x44, 0x88, 0xd, 0x1a, 0x34, 0x68, 0xd0, 0xbd, 0x67, 0xce, 0x81,
    0x1f, 0x3e, 0x7c, 0xf8, 0xed, 0xc7, 0x93, 0x3b, 0x76, 0xec, 0xc5, 0x97, 0x33, 0x66, 0xcc, 0x85,
    0x17, 0x2e, 0x5c, 0xb8, 0x6d, 0xda, 0xa9, 0x4f, 0x9e, 0x21, 0x42, 0x84, 0x15, 0x2a, 0x54, 0xa8,
    0x4d, 0x9a, 0x29, 0x52, 0xa4, 0x55, 0xaa, 0x49, 0x92, 0x39, 0x72, 0xe4, 0xd5, 0xb7, 0x73, 0xe6,
    0xd1, 0xbf, 0x63, 0xc6, 0x91, 0x3f, 0x7e, 0xfc, 0xe5, 0xd7, 0xb3, 0x7b, 0xf6, 0xf1, 0xff, 0xe3,
    0xdb, 0xab, 0x4b, 0x96, 0x31, 0x62, 0xc4, 0x95, 0x37, 0x6e, 0xdc, 0xa5, 0x57, 0xae, 0x41, 0x82,
    0x19, 0x32, 0x64, 0xc8, 0x8d, 0x7, 0xe, 0x1c, 0x38, 0x70, 0xe0, 0xdd, 0xa7, 0x53, 0xa6, 0x51,
    0xa2, 0x59, 0xb2, 0x79, 0xf2, 0xf9, 0xef, 0xc3, 0x9b, 0x2b, 0x56, 0xac, 0x45, 0x8a, 0x9, 0x12,
    0x24, 0x48, 0x90, 0x3d, 0x7a, 0xf4, 0xf5, 0xf7, 0xf3, 0xfb, 0xeb, 0xcb, 0x8b, 0xb, 0x16, 0x2c,
    0x58, 0xb0, 0x7d, 0xfa, 0xe9, 0xcf, 0x83, 0x1b, 0x36, 0x6c, 0xd8, 0xad, 0x47, 0x8e, 0x1, 0x2,
    // second half (repeat)
    0x4, 0x8, 0x10, 0x20, 0x40, 0x80, 0x1d, 0x3a, 0x74, 0xe8, 0xcd, 0x87, 0x13, 0x26, 0x4c, 0x98,
    0x2d, 0x5a, 0xb4, 0x75, 0xea, 0xc9, 0x8f, 0x3, 0x6, 0xc, 0x18, 0x30, 0x60, 0xc0, 0x9d, 0x27,
    0x4e, 0x9c, 0x25, 0x4a, 0x94, 0x35, 0x6a, 0xd4, 0xb5, 0x77, 0xee, 0xc1, 0x9f, 0x23, 0x46, 0x8c,
    0x5, 0xa, 0x14, 0x28, 0x50, 0xa0, 0x5d, 0xba, 0x69, 0xd2, 0xb9, 0x6f, 0xde, 0xa1, 0x5f, 0xbe,
    0x61, 0xc2, 0x99, 0x2f, 0x5e, 0xbc, 0x65, 0xca, 0x89, 0xf, 0x1e, 0x3c, 0x78, 0xf0, 0xfd, 0xe7,
    0xd3, 0xbb, 0x6b, 0xd6, 0xb1, 0x7f, 0xfe, 0xe1, 0xdf, 0xa3, 0x5b, 0xb6, 0x71, 0xe2, 0xd9, 0xaf,
    0x43, 0x86, 0x11, 0x22, 0x44, 0x88, 0xd, 0x1a, 0x34, 0x68, 0xd0, 0xbd, 0x67, 0xce, 0x81, 0x1f,
    0x3e, 0x7c, 0xf8, 0xed, 0xc7, 0x93, 0x3b, 0x76, 0xec, 0xc5, 0x97, 0x33, 0x66, 0xcc, 0x85, 0x17,
    0x2e, 0x5c, 0xb8, 0x6d, 0xda, 0xa9, 0x4f, 0x9e, 0x21, 0x42, 0x84, 0x15, 0x2a, 0x54, 0xa8, 0x4d,
    0x9a, 0x29, 0x52, 0xa4, 0x55, 0xaa, 0x49, 0x92, 0x39, 0x72, 0xe4, 0xd5, 0xb7, 0x73, 0xe6, 0xd1,
    0xbf, 0x63, 0xc6, 0x91, 0x3f, 0x7e, 0xfc, 0xe5, 0xd7, 0xb3, 0x7b, 0xf6, 0xf1, 0xff, 0xe3, 0xdb,
    0xab, 0x4b, 0x96, 0x31, 0x62, 0xc4, 0x95, 0x37, 0x6e, 0xdc, 0xa5, 0x57, 0xae, 0x41, 0x82, 0x19,
    0x32, 0x64, 0xc8, 0x8d, 0x7, 0xe, 0x1c, 0x38, 0x70, 0xe0, 0xdd, 0xa7, 0x53, 0xa6, 0x51, 0xa2,
    0x59, 0xb2, 0x79, 0xf2, 0xf9, 0xef, 0xc3, 0x9b, 0x2b, 0x56, 0xac, 0x45, 0x8a, 0x9, 0x12, 0x24,
    0x48, 0x90, 0x3d, 0x7a, 0xf4, 0xf5, 0xf7, 0xf3, 0xfb, 0xeb, 0xcb, 0x8b, 0xb, 0x16, 0x2c, 0x58,
    0xb0, 0x7d, 0xfa, 0xe9, 0xcf, 0x83, 0x1b, 0x36, 0x6c, 0xd8, 0xad, 0x47, 0x8e, 0x1, 0x2,
];

/// GF(2^8) logarithm table (256 entries). log[0] = 0 by convention.
#[rustfmt::skip]
const GF_LOG: [u8; 256] = [
    0x0, 0x0, 0x1, 0x19, 0x2, 0x32, 0x1a, 0xc6, 0x3, 0xdf, 0x33, 0xee, 0x1b, 0x68, 0xc7, 0x4b, 0x4,
    0x64, 0xe0, 0xe, 0x34, 0x8d, 0xef, 0x81, 0x1c, 0xc1, 0x69, 0xf8, 0xc8, 0x8, 0x4c, 0x71, 0x5,
    0x8a, 0x65, 0x2f, 0xe1, 0x24, 0xf, 0x21, 0x35, 0x93, 0x8e, 0xda, 0xf0, 0x12, 0x82, 0x45, 0x1d,
    0xb5, 0xc2, 0x7d, 0x6a, 0x27, 0xf9, 0xb9, 0xc9, 0x9a, 0x9, 0x78, 0x4d, 0xe4, 0x72, 0xa6, 0x6,
    0xbf, 0x8b, 0x62, 0x66, 0xdd, 0x30, 0xfd, 0xe2, 0x98, 0x25, 0xb3, 0x10, 0x91, 0x22, 0x88, 0x36,
    0xd0, 0x94, 0xce, 0x8f, 0x96, 0xdb, 0xbd, 0xf1, 0xd2, 0x13, 0x5c, 0x83, 0x38, 0x46, 0x40, 0x1e,
    0x42, 0xb6, 0xa3, 0xc3, 0x48, 0x7e, 0x6e, 0x6b, 0x3a, 0x28, 0x54, 0xfa, 0x85, 0xba, 0x3d, 0xca,
    0x5e, 0x9b, 0x9f, 0xa, 0x15, 0x79, 0x2b, 0x4e, 0xd4, 0xe5, 0xac, 0x73, 0xf3, 0xa7, 0x57, 0x7,
    0x70, 0xc0, 0xf7, 0x8c, 0x80, 0x63, 0xd, 0x67, 0x4a, 0xde, 0xed, 0x31, 0xc5, 0xfe, 0x18, 0xe3,
    0xa5, 0x99, 0x77, 0x26, 0xb8, 0xb4, 0x7c, 0x11, 0x44, 0x92, 0xd9, 0x23, 0x20, 0x89, 0x2e, 0x37,
    0x3f, 0xd1, 0x5b, 0x95, 0xbc, 0xcf, 0xcd, 0x90, 0x87, 0x97, 0xb2, 0xdc, 0xfc, 0xbe, 0x61, 0xf2,
    0x56, 0xd3, 0xab, 0x14, 0x2a, 0x5d, 0x9e, 0x84, 0x3c, 0x39, 0x53, 0x47, 0x6d, 0x41, 0xa2, 0x1f,
    0x2d, 0x43, 0xd8, 0xb7, 0x7b, 0xa4, 0x76, 0xc4, 0x17, 0x49, 0xec, 0x7f, 0xc, 0x6f, 0xf6, 0x6c,
    0xa1, 0x3b, 0x52, 0x29, 0x9d, 0x55, 0xaa, 0xfb, 0x60, 0x86, 0xb1, 0xbb, 0xcc, 0x3e, 0x5a, 0xcb,
    0x59, 0x5f, 0xb0, 0x9c, 0xa9, 0xa0, 0x51, 0xb, 0xf5, 0x16, 0xeb, 0x7a, 0x75, 0x2c, 0xd7, 0x4f,
    0xae, 0xd5, 0xe9, 0xe6, 0xe7, 0xad, 0xe8, 0x74, 0xd6, 0xf4, 0xea, 0xa8, 0x50, 0x58, 0xaf,
];

// --- GF(2^8) arithmetic ---

#[inline]
fn gf_mul(x: u8, y: u8) -> u8 {
    if x == 0 || y == 0 {
        0
    } else {
        GF_EXP[GF_LOG[x as usize] as usize + GF_LOG[y as usize] as usize]
    }
}

#[inline]
fn gf_div(x: u8, y: u8) -> u8 {
    debug_assert!(y != 0);
    if x == 0 {
        return 0;
    }
    GF_EXP[(GF_LOG[x as usize] as usize + 255 - GF_LOG[y as usize] as usize) % 255]
}

#[inline]
fn gf_pow(x: u8, power: isize) -> u8 {
    let mut i = GF_LOG[x as usize] as isize * power;
    i %= 255;
    if i < 0 {
        i += 255;
    }
    GF_EXP[i as usize]
}

#[inline]
fn gf_inverse(x: u8) -> u8 {
    GF_EXP[255 - GF_LOG[x as usize] as usize]
}

// --- Polynomial operations (dynamic Vec-based) ---

fn poly_scale(p: &[u8], x: u8) -> Vec<u8> {
    p.iter().map(|&c| gf_mul(c, x)).collect()
}

fn poly_add(p: &[u8], q: &[u8]) -> Vec<u8> {
    let len = p.len().max(q.len());
    let mut result = vec![0u8; len];
    for (i, &v) in p.iter().enumerate() {
        result[i + len - p.len()] = v;
    }
    for (i, &v) in q.iter().enumerate() {
        result[i + len - q.len()] ^= v;
    }
    result
}

fn poly_mul(p: &[u8], q: &[u8]) -> Vec<u8> {
    let mut result = vec![0u8; p.len() + q.len() - 1];
    for (j, &qv) in q.iter().enumerate() {
        for (i, &pv) in p.iter().enumerate() {
            result[i + j] ^= gf_mul(pv, qv);
        }
    }
    result
}

fn poly_div(p: &[u8], q: &[u8]) -> Vec<u8> {
    let mut result = p.to_vec();
    for i in 0..(p.len() - (q.len() - 1)) {
        let coef = result[i];
        if coef != 0 {
            for j in 1..q.len() {
                if q[j] != 0 {
                    result[i + j] ^= gf_mul(q[j], coef);
                }
            }
        }
    }
    let sep = p.len() - (q.len() - 1);
    result[sep..].to_vec()
}

fn poly_eval(p: &[u8], x: u8) -> u8 {
    let mut y = p[0];
    for &c in &p[1..] {
        y = gf_mul(y, x) ^ c;
    }
    y
}

// --- Generator polynomial ---

fn generator_poly(ecc_length: usize) -> Vec<u8> {
    let mut gen = vec![1u8];
    for i in 0..ecc_length {
        let mulp = [1u8, gf_pow(2, i as isize)];
        gen = poly_mul(&gen, &mulp);
    }
    gen
}

// --- Syndromes ---

fn calc_syndromes(msg: &[u8], ecc_length: usize) -> Vec<u8> {
    let mut synd = vec![0u8; ecc_length + 1];
    synd[0] = 0;
    for i in 1..=ecc_length {
        synd[i] = poly_eval(msg, gf_pow(2, (i - 1) as isize));
    }
    synd
}

// --- Error locator (Berlekamp-Massey) ---

fn find_error_locator(synd: &[u8], ecc_length: usize, erase_count: usize) -> Option<Vec<u8>> {
    let mut err_loc = vec![1u8];
    let mut old_loc = vec![1u8];

    let synd_shift = if synd.len() > ecc_length {
        synd.len() - ecc_length
    } else {
        0
    };

    for i in 0..(ecc_length - erase_count) {
        let k = i + synd_shift;

        let mut delta = synd[k];
        for j in 1..err_loc.len() {
            let index = err_loc.len() - j - 1;
            delta ^= gf_mul(err_loc[index], synd[k - j]);
        }

        old_loc.push(0);

        if delta != 0 {
            if old_loc.len() > err_loc.len() {
                let temp = poly_scale(&old_loc, delta);
                old_loc = poly_scale(&err_loc, gf_inverse(delta));
                err_loc = temp;
            }
            let temp = poly_scale(&old_loc, delta);
            err_loc = poly_add(&err_loc, &temp);
        }
    }

    // strip leading zeros
    let mut shift = 0;
    while shift < err_loc.len() && err_loc[shift] == 0 {
        shift += 1;
    }
    let err_loc = err_loc[shift..].to_vec();

    let errs = err_loc.len() - 1;
    if (errs - erase_count) * 2 + erase_count > ecc_length {
        return None;
    }

    Some(err_loc)
}

// --- Find error positions (Chien search) ---

fn find_errors(error_loc: &[u8], msg_size: usize) -> Option<Vec<usize>> {
    let errs = error_loc.len() - 1;
    let mut err_pos = Vec::new();

    for i in 0..msg_size {
        if poly_eval(error_loc, gf_pow(2, i as isize)) == 0 {
            err_pos.push(msg_size - 1 - i);
        }
    }

    if err_pos.len() != errs {
        return None;
    }

    Some(err_pos)
}

// --- Errata locator ---

fn find_errata_locator(epos: &[usize], msg_len: usize) -> Vec<u8> {
    let c_pos: Vec<usize> = epos.iter().map(|&p| msg_len - 1 - p).collect();
    let mut errata_loc = vec![1u8];
    for &p in &c_pos {
        let addp = [gf_pow(2, p as isize), 0];
        let mulp = [1u8];
        let apol = poly_add(&mulp, &addp);
        errata_loc = poly_mul(&errata_loc, &apol);
    }
    errata_loc
}

// --- Error evaluator ---

fn find_error_evaluator(synd: &[u8], errata_loc: &[u8], ecc_len: usize) -> Vec<u8> {
    let mulp = poly_mul(synd, errata_loc);
    let mut divisor = vec![0u8; ecc_len + 2];
    divisor[0] = 1;
    poly_div(&mulp, &divisor)
}

// --- Correct errata ---

fn correct_errata(msg_in: &[u8], synd: &[u8], err_pos: &[usize]) -> Vec<u8> {
    let msg_len = msg_in.len();

    let c_pos: Vec<usize> = err_pos.iter().map(|&p| msg_len - 1 - p).collect();

    let errata_loc = find_errata_locator(err_pos, msg_len);

    // reverse syndromes
    let rsynd: Vec<u8> = synd.iter().rev().copied().collect();

    // error evaluator
    let re_eval = find_error_evaluator(&rsynd, &errata_loc, errata_loc.len() - 1);

    // X: error positions as GF elements
    let x_vals: Vec<u8> = c_pos
        .iter()
        .map(|&p| {
            let l = 255isize - p as isize;
            gf_pow(2, -l)
        })
        .collect();

    // Magnitude polynomial
    let mut e = vec![0u8; msg_len];

    for i in 0..x_vals.len() {
        let xi_inv = gf_inverse(x_vals[i]);

        // err_loc_prime
        let mut err_loc_prime = 1u8;
        for j in 0..x_vals.len() {
            if j != i {
                let val = 1 ^ gf_mul(xi_inv, x_vals[j]);
                err_loc_prime = gf_mul(err_loc_prime, val);
            }
        }

        let y = gf_mul(gf_pow(x_vals[i], 1), poly_eval(&re_eval, xi_inv));

        e[err_pos[i]] = gf_div(y, err_loc_prime);
    }

    poly_add(msg_in, &e)
}

/// Forney syndromes for error detection after erasure removal.
fn calc_forney_syndromes(
    synd: &[u8],
    erasures_pos: &[usize],
    msg_in_size: usize,
) -> Vec<u8> {
    let erase_pos_reversed: Vec<usize> = erasures_pos
        .iter()
        .map(|&p| msg_in_size - 1 - p)
        .collect();

    let mut forney_synd = synd[1..].to_vec();

    for &p in &erase_pos_reversed {
        let x = gf_pow(2, p as isize);
        for j in 0..forney_synd.len().saturating_sub(1) {
            forney_synd[j] = gf_mul(forney_synd[j], x) ^ forney_synd[j + 1];
        }
    }

    forney_synd
}

/// Reed-Solomon encoder/decoder.
pub struct ReedSolomon {
    msg_length: usize,
    ecc_length: usize,
    generator: Vec<u8>,
}

impl ReedSolomon {
    /// Create a new RS codec with the given message and ECC lengths.
    pub fn new(msg_length: usize, ecc_length: usize) -> Self {
        let generator = generator_poly(ecc_length);
        Self {
            msg_length,
            ecc_length,
            generator,
        }
    }

    /// Encode: appends ECC bytes to the message.
    /// Returns a vector of `msg_length + ecc_length` bytes.
    pub fn encode(&self, msg: &[u8]) -> Vec<u8> {
        assert_eq!(msg.len(), self.msg_length);
        assert!(self.msg_length + self.ecc_length < 256);

        let mut msg_out = vec![0u8; self.msg_length + self.ecc_length];
        msg_out[..self.msg_length].copy_from_slice(msg);

        for i in 0..self.msg_length {
            let coef = msg_out[i];
            if coef != 0 {
                for j in 1..self.generator.len() {
                    msg_out[i + j] ^= gf_mul(self.generator[j], coef);
                }
            }
        }

        // The first msg_length bytes get overwritten by the division;
        // restore the original message.
        msg_out[..self.msg_length].copy_from_slice(msg);

        msg_out
    }

    /// Encode block: returns only the ECC bytes (not the message).
    pub fn encode_block(&self, msg: &[u8]) -> Vec<u8> {
        let full = self.encode(msg);
        full[self.msg_length..].to_vec()
    }

    /// Decode: takes `msg_length + ecc_length` bytes, returns decoded message.
    /// Returns `None` if correction fails.
    pub fn decode(&self, encoded: &[u8]) -> Option<Vec<u8>> {
        assert_eq!(encoded.len(), self.msg_length + self.ecc_length);

        let src_len = self.msg_length + self.ecc_length;

        // Calculate syndromes
        let synd = calc_syndromes(encoded, self.ecc_length);

        // Check for errors
        let has_errors = synd.iter().any(|&s| s != 0);
        if !has_errors {
            return Some(encoded[..self.msg_length].to_vec());
        }

        // No erasures in our use case
        let erase_pos: Vec<usize> = Vec::new();

        let forney = calc_forney_syndromes(&synd, &erase_pos, src_len);

        let error_loc = find_error_locator(&forney, self.ecc_length, erase_pos.len())?;

        // reverse error locator
        let reloc: Vec<u8> = error_loc.iter().rev().copied().collect();

        let err = find_errors(&reloc, src_len)?;

        if err.is_empty() {
            return None;
        }

        // combine erasure positions with found errors
        let mut all_pos = erase_pos;
        all_pos.extend_from_slice(&err);

        let corrected = correct_errata(encoded, &synd, &all_pos);

        Some(corrected[..self.msg_length].to_vec())
    }

    /// Decode block: takes separate message and ecc slices.
    /// Returns decoded message or None if correction fails.
    pub fn decode_block(&self, msg: &[u8], ecc: &[u8]) -> Option<Vec<u8>> {
        let mut encoded = Vec::with_capacity(self.msg_length + self.ecc_length);
        encoded.extend_from_slice(msg);
        encoded.extend_from_slice(ecc);
        self.decode(&encoded)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gf_mul_basic() {
        assert_eq!(gf_mul(0, 5), 0);
        assert_eq!(gf_mul(5, 0), 0);
        assert_eq!(gf_mul(1, 1), 1);
        assert_eq!(gf_mul(2, 2), 4);
    }

    #[test]
    fn test_gf_div_basic() {
        for x in 1..=255u8 {
            assert_eq!(gf_div(x, x), 1);
        }
    }

    #[test]
    fn test_gf_pow_basic() {
        assert_eq!(gf_pow(2, 0), 1);
        assert_eq!(gf_pow(2, 1), 2);
        assert_eq!(gf_pow(2, 8), 0x1d); // 2^8 mod p(x) = 0x1d
    }

    #[test]
    fn test_rs_encode_decode_roundtrip() {
        let rs = ReedSolomon::new(3, 2);
        let msg = [0x48, 0x65, 0x6c]; // "Hel"
        let encoded = rs.encode(&msg);
        assert_eq!(encoded.len(), 5);

        let decoded = rs.decode(&encoded).unwrap();
        assert_eq!(&decoded, &msg);
    }

    #[test]
    fn test_rs_correct_single_error() {
        let rs = ReedSolomon::new(3, 4);
        let msg = [0x01, 0x02, 0x03];
        let encoded = rs.encode(&msg);

        let mut corrupted = encoded.clone();
        corrupted[1] ^= 0xFF;

        let decoded = rs.decode(&corrupted).unwrap();
        assert_eq!(&decoded, &msg);
    }

    #[test]
    fn test_rs_length_encoding() {
        // This matches ggwave's RS(1, 2) for the length byte
        let rs = ReedSolomon::new(1, 2);
        let msg = [5u8]; // payload length = 5
        let encoded = rs.encode(&msg);
        assert_eq!(encoded.len(), 3);

        let decoded = rs.decode(&encoded).unwrap();
        assert_eq!(&decoded, &msg);
    }

    #[test]
    fn test_rs_larger_message() {
        let msg_len = 20;
        let ecc_len = 8; // 2*(20/5) = 8
        let rs = ReedSolomon::new(msg_len, ecc_len);
        let msg: Vec<u8> = (0..msg_len as u8).collect();
        let encoded = rs.encode(&msg);

        // corrupt two bytes
        let mut corrupted = encoded;
        corrupted[3] ^= 0xAB;
        corrupted[10] ^= 0xCD;

        let decoded = rs.decode(&corrupted).unwrap();
        assert_eq!(&decoded, &msg[..]);
    }
}
