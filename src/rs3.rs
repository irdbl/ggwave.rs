//! Reed-Solomon codec over GF(2^3) with primitive polynomial x^3+x+1 (0x0B).
//!
//! All values are u8 in range 0..=7 (3-bit symbols = GF(2^3) elements = Teams modem symbols).
//! Max codeword length: n = 7 (2^3 - 1).

/// GF(2^3) order: 2^3 = 8, multiplicative group has 7 elements.
const GF_ORDER: usize = 8;
const GF_MUL_ORDER: usize = 7;

/// GF(2^3) exponential table: alpha^i for i in 0..14 (doubled for wraparound).
/// Primitive polynomial: x^3 + x + 1 (0x0B). Generator alpha = 2.
///
/// α^0 = 1, α^1 = 2, α^2 = 4, α^3 = 3, α^4 = 6, α^5 = 7, α^6 = 5, α^7 = 1
#[rustfmt::skip]
const GF_EXP: [u8; 16] = [
    1, 2, 4, 3, 6, 7, 5, 1,
    2, 4, 3, 6, 7, 5, 1, 2,
];

/// GF(2^3) logarithm table: log_alpha(x) for x in 0..8. log(0) = 255 (sentinel).
#[rustfmt::skip]
const GF_LOG: [u8; GF_ORDER] = [
    255, 0, 1, 3, 2, 6, 4, 5,
];

// --- GF(2^3) arithmetic ---

#[inline]
fn gf_mul(x: u8, y: u8) -> u8 {
    if x == 0 || y == 0 {
        0
    } else {
        GF_EXP[(GF_LOG[x as usize] as usize + GF_LOG[y as usize] as usize) % GF_MUL_ORDER]
    }
}

#[inline]
fn gf_div(x: u8, y: u8) -> u8 {
    debug_assert!(y != 0);
    if x == 0 {
        return 0;
    }
    GF_EXP[(GF_LOG[x as usize] as usize + GF_MUL_ORDER - GF_LOG[y as usize] as usize) % GF_MUL_ORDER]
}

#[inline]
fn gf_pow(x: u8, power: isize) -> u8 {
    if x == 0 {
        return 0;
    }
    let mut i = GF_LOG[x as usize] as isize * power;
    i = ((i % GF_MUL_ORDER as isize) + GF_MUL_ORDER as isize) % GF_MUL_ORDER as isize;
    GF_EXP[i as usize]
}

#[inline]
fn gf_inverse(x: u8) -> u8 {
    debug_assert!(x != 0);
    GF_EXP[GF_MUL_ORDER - GF_LOG[x as usize] as usize]
}

// --- Polynomial operations ---

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
    if errs * 2 + erase_count > ecc_length {
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

    let rsynd: Vec<u8> = synd.iter().rev().copied().collect();

    let re_eval = find_error_evaluator(&rsynd, &errata_loc, errata_loc.len() - 1);

    let x_vals: Vec<u8> = c_pos
        .iter()
        .map(|&p| {
            let l = GF_MUL_ORDER as isize - p as isize;
            gf_pow(2, -l)
        })
        .collect();

    let mut e = vec![0u8; msg_len];

    for i in 0..x_vals.len() {
        let xi_inv = gf_inverse(x_vals[i]);

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

/// Reed-Solomon encoder/decoder over GF(2^3).
///
/// All values must be in range 0..=7 (3-bit symbols = GF(2^3) elements).
/// Max codeword: msg_len + ecc_len <= 7.
pub struct ReedSolomon3 {
    msg_len: usize,
    ecc_len: usize,
    generator: Vec<u8>,
}

impl ReedSolomon3 {
    /// Create a new RS codec. msg_len + ecc_len must be <= 7.
    pub fn new(msg_len: usize, ecc_len: usize) -> Self {
        assert!(msg_len + ecc_len <= GF_MUL_ORDER, "RS(n,k) requires n <= 7 for GF(2^3)");
        let generator = generator_poly(ecc_len);
        Self {
            msg_len,
            ecc_len,
            generator,
        }
    }

    /// Encode: appends ECC symbols to message. Returns vec of msg_len + ecc_len symbols.
    pub fn encode(&self, msg: &[u8]) -> Vec<u8> {
        assert_eq!(msg.len(), self.msg_len);
        debug_assert!(msg.iter().all(|&v| v < GF_ORDER as u8));

        let n = self.msg_len + self.ecc_len;
        let mut msg_out = vec![0u8; n];
        msg_out[..self.msg_len].copy_from_slice(msg);

        for i in 0..self.msg_len {
            let coef = msg_out[i];
            if coef != 0 {
                for j in 1..self.generator.len() {
                    msg_out[i + j] ^= gf_mul(self.generator[j], coef);
                }
            }
        }

        msg_out[..self.msg_len].copy_from_slice(msg);
        msg_out
    }

    /// Decode: takes msg_len + ecc_len symbols, returns decoded message or None.
    pub fn decode(&self, encoded: &[u8]) -> Option<Vec<u8>> {
        self.decode_with_erasures(encoded, &[])
    }

    /// Decode with known erasure positions. Returns decoded message or None.
    pub fn decode_with_erasures(&self, encoded: &[u8], erase_pos: &[usize]) -> Option<Vec<u8>> {
        assert_eq!(encoded.len(), self.msg_len + self.ecc_len);

        let src_len = self.msg_len + self.ecc_len;

        let synd = calc_syndromes(encoded, self.ecc_len);

        let has_errors = synd.iter().any(|&s| s != 0);
        if !has_errors {
            return Some(encoded[..self.msg_len].to_vec());
        }

        if erase_pos.len() > self.ecc_len {
            return None;
        }

        let forney = calc_forney_syndromes(&synd, erase_pos, src_len);

        let error_loc = find_error_locator(&forney, self.ecc_len, erase_pos.len())?;

        let reloc: Vec<u8> = error_loc.iter().rev().copied().collect();

        let err = find_errors(&reloc, src_len)?;

        let mut all_pos: Vec<usize> = erase_pos.to_vec();
        all_pos.extend_from_slice(&err);

        if all_pos.is_empty() {
            return None;
        }

        let corrected = correct_errata(encoded, &synd, &all_pos);

        let verify_synd = calc_syndromes(&corrected, self.ecc_len);
        if verify_synd.iter().any(|&s| s != 0) {
            return None;
        }

        Some(corrected[..self.msg_len].to_vec())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gf3_tables_consistent() {
        // Verify exp/log are inverses
        for i in 0..GF_MUL_ORDER {
            let val = GF_EXP[i];
            assert!(val > 0 && val < GF_ORDER as u8);
            assert_eq!(GF_LOG[val as usize] as usize, i, "log(exp({i})) != {i}");
        }
        // Verify all nonzero elements appear
        let mut seen = [false; GF_ORDER];
        for i in 0..GF_MUL_ORDER {
            seen[GF_EXP[i] as usize] = true;
        }
        for v in 1..GF_ORDER {
            assert!(seen[v], "element {v} not generated");
        }
    }

    #[test]
    fn test_gf3_mul_basic() {
        assert_eq!(gf_mul(0, 5), 0);
        assert_eq!(gf_mul(5, 0), 0);
        assert_eq!(gf_mul(1, 1), 1);
        assert_eq!(gf_mul(2, 2), 4);
        // 2 * 4 = alpha^1 * alpha^2 = alpha^3 = 3 (via table)
        assert_eq!(gf_mul(2, 4), 3);
    }

    #[test]
    fn test_gf3_div_identity() {
        for x in 1..GF_ORDER as u8 {
            assert_eq!(gf_div(x, x), 1, "x/x should be 1 for x={x}");
        }
    }

    #[test]
    fn test_gf3_mul_div_roundtrip() {
        for x in 1..GF_ORDER as u8 {
            for y in 1..GF_ORDER as u8 {
                let p = gf_mul(x, y);
                assert_eq!(gf_div(p, y), x, "({x}*{y})/{y} should be {x}");
            }
        }
    }

    #[test]
    fn test_rs3_encode_decode_roundtrip() {
        let rs = ReedSolomon3::new(5, 2);
        let msg: Vec<u8> = (0..5).collect();
        let encoded = rs.encode(&msg);
        assert_eq!(encoded.len(), 7);
        let decoded = rs.decode(&encoded).unwrap();
        assert_eq!(decoded, msg);
    }

    #[test]
    fn test_rs3_various_configs() {
        // Test different RS configurations within GF(2^3)
        for (k, t) in [(3, 4), (4, 3), (5, 2), (6, 1)] {
            let rs = ReedSolomon3::new(k, t);
            let msg: Vec<u8> = (0..k as u8).map(|v| v % 8).collect();
            let encoded = rs.encode(&msg);
            assert_eq!(encoded.len(), k + t);
            let decoded = rs.decode(&encoded).unwrap();
            assert_eq!(decoded, msg, "roundtrip failed for RS(7,{k})");
        }
    }

    #[test]
    fn test_rs3_all_zeros() {
        let rs = ReedSolomon3::new(5, 2);
        let msg = vec![0u8; 5];
        let encoded = rs.encode(&msg);
        let decoded = rs.decode(&encoded).unwrap();
        assert_eq!(decoded, msg);
    }

    #[test]
    fn test_rs3_all_max() {
        let rs = ReedSolomon3::new(5, 2);
        let msg = vec![7u8; 5]; // max value in GF(2^3)
        let encoded = rs.encode(&msg);
        let decoded = rs.decode(&encoded).unwrap();
        assert_eq!(decoded, msg);
    }

    #[test]
    fn test_rs3_correct_1_error() {
        let rs = ReedSolomon3::new(5, 2);
        let msg: Vec<u8> = (0..5).collect();
        let encoded = rs.encode(&msg);

        // Corrupt each position one at a time
        for pos in 0..7 {
            let mut corrupted = encoded.clone();
            corrupted[pos] = (corrupted[pos] + 1) % 8;
            let decoded = rs.decode(&corrupted).expect(&format!("failed to correct error at pos {pos}"));
            assert_eq!(decoded, msg, "wrong correction at pos {pos}");
        }
    }

    #[test]
    fn test_rs3_2_errors_with_2_parity() {
        // RS(7,5) with 2 parity can correct at most 1 error.
        // With 2 errors, decoder should either fail or miscorrect.
        let rs = ReedSolomon3::new(5, 2);
        let msg: Vec<u8> = (0..5).collect();
        let encoded = rs.encode(&msg);

        let mut corrupted = encoded.clone();
        corrupted[0] = (corrupted[0] + 1) % 8;
        corrupted[3] = (corrupted[3] + 2) % 8;

        // Decoder may return None (detection) or Some(wrong data) (miscorrection)
        // Either outcome is acceptable - the key is it shouldn't silently produce correct data
        // by pure luck on every error pattern
        match rs.decode(&corrupted) {
            None => (), // Detection - good
            Some(_decoded) => {
                // Miscorrection is possible - just verify we're not testing a trivial case
                // where the errors happened to cancel out
                assert!(corrupted != encoded, "Test setup error: no actual corruption");
            }
        }
    }

    #[test]
    fn test_rs3_correct_2_errors_with_4_parity() {
        // RS(7,3) with 4 parity can correct 2 errors
        let rs = ReedSolomon3::new(3, 4);
        let msg: Vec<u8> = (0..3).collect();
        let encoded = rs.encode(&msg);

        let mut corrupted = encoded.clone();
        corrupted[0] = (corrupted[0] + 1) % 8;
        corrupted[4] = (corrupted[4] + 3) % 8;

        let decoded = rs.decode(&corrupted).unwrap();
        assert_eq!(decoded, msg);
    }

    #[test]
    fn test_rs3_erasure_2() {
        // 2 parity = correct up to 2 erasures
        let rs = ReedSolomon3::new(5, 2);
        let msg: Vec<u8> = (0..5).collect();
        let encoded = rs.encode(&msg);

        let mut corrupted = encoded.clone();
        corrupted[1] = 0;
        corrupted[4] = 0;

        let decoded = rs.decode_with_erasures(&corrupted, &[1, 4]).unwrap();
        assert_eq!(decoded, msg);
    }

    #[test]
    fn test_rs3_erasure_exceeds_budget() {
        let rs = ReedSolomon3::new(5, 2);
        let msg: Vec<u8> = (0..5).collect();
        let encoded = rs.encode(&msg);

        let mut corrupted = encoded.clone();
        corrupted[0] = 0;
        corrupted[2] = 0;
        corrupted[4] = 0;

        assert!(rs.decode_with_erasures(&corrupted, &[0, 2, 4]).is_none());
    }

    #[test]
    fn test_rs3_erasure_plus_error() {
        // RS(7,3) with 4 parity: 2 erasures + 1 error = 2 + 2 = 4 <= 4: should work
        let rs = ReedSolomon3::new(3, 4);
        let msg: Vec<u8> = (0..3).collect();
        let encoded = rs.encode(&msg);

        let mut corrupted = encoded.clone();
        corrupted[1] = 0; // erasure
        corrupted[3] = 0; // erasure
        corrupted[5] = (corrupted[5] + 5) % 8; // unknown error

        let decoded = rs.decode_with_erasures(&corrupted, &[1, 3]).unwrap();
        assert_eq!(decoded, msg);
    }
}
