// SPDX-License-Identifier: LGPL-2.1-or-later
//! In-process ONNX implementation, for the scoring host and offline tools.
use super::D512Error;

#[derive(Debug, Clone, PartialEq)]
struct Input {
    ids: Vec<i64>,
    bases: Vec<i64>,
    tones: Vec<i64>,
    types: Vec<i64>,
    positions: Vec<i64>,
    mask: Vec<bool>,
    proj_pos: Vec<i64>,
    target_node: Vec<i64>,
    target_ids: Vec<i64>,
}

fn encode(context: &[i64], sounds: &[(i64, i64)], candidate: &[i64]) -> Input {
    let context = &context[context.len().saturating_sub(64)..];
    let h = context.len() + 2;
    let k = sounds.len();
    let n = candidate.len();
    let out = h + k;
    let len = out + n;
    let mut ids = vec![1];
    ids.extend_from_slice(context);
    ids.push(2);
    ids.extend(vec![0; k]);
    ids.push(3);
    ids.extend_from_slice(&candidate[..n - 1]);
    let mut bases = vec![0; h];
    bases.extend(sounds.iter().map(|s| s.0));
    bases.extend(vec![0; n]);
    let mut tones = vec![0; h];
    tones.extend(sounds.iter().map(|s| s.1));
    tones.extend(vec![0; n]);
    let mut types = vec![0; h];
    types.extend(vec![1; k]);
    types.push(0);
    types.extend(vec![2; n - 1]);
    Input {
        ids,
        bases,
        tones,
        types,
        positions: (0..len as i64).collect(),
        mask: (0..len)
            .flat_map(|r| (0..len).map(move |c| c <= r))
            .collect(),
        proj_pos: (out as i64..len as i64).collect(),
        target_node: (0..n as i64).collect(),
        target_ids: candidate.iter().map(|c| c - 6).collect(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn phonetic_tensor_protocol_and_context_tail() {
        let x = encode(&[8, 9], &[(2, 1), (3, 2)], &[10, 11]);
        assert_eq!(x.ids, [1, 8, 9, 2, 0, 0, 3, 10]);
        assert_eq!(x.bases, [0, 0, 0, 0, 2, 3, 0, 0]);
        assert_eq!(x.tones, [0, 0, 0, 0, 1, 2, 0, 0]);
        assert_eq!(x.types, [0, 0, 0, 0, 1, 1, 0, 2]);
        assert_eq!(x.positions, (0..8).collect::<Vec<_>>());
        assert_eq!(x.proj_pos, [6, 7]);
        assert_eq!(x.target_node, [0, 1]);
        assert_eq!(x.target_ids, [4, 5]);
        assert_eq!(
            x.mask,
            (0..8)
                .flat_map(|r| (0..8).map(move |c| c <= r))
                .collect::<Vec<_>>()
        );
        let long = encode(&vec![8; 70], &[(2, 1)], &[9]);
        assert_eq!(long.ids.len(), 68);
    }

    #[test]
    fn text_mode_has_no_reading_slots() {
        let x = encode(&[8], &[], &[9]);
        assert_eq!(x.ids, [1, 8, 2, 3]);
        assert_eq!(x.bases, [0, 0, 0, 0]);
        assert_eq!(x.tones, [0, 0, 0, 0]);
        assert_eq!(x.types, [0, 0, 0, 0]);
        assert_eq!(x.proj_pos, [3]);
    }
}
