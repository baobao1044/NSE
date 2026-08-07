//! Character-level tokenizer for the Toy LM POC.
//!
//! Maps each byte/char of the training corpus to a token id. Simple and
//! deterministic, no BPE — sufficient to validate the NSE pipeline.

use std::collections::HashMap;

/// A byte-level char tokenizer.
#[derive(Debug, Clone)]
pub struct Tokenizer {
    /// token id for each byte value 0..=255
    pub id_by_byte: [u32; 256],
    /// byte for each token id (vocab_size entries)
    pub byte_by_id: Vec<u8>,
    pub vocab_size: usize,
}

impl Tokenizer {
    /// Build a tokenizer over the full byte alphabet (vocab_size = 256).
    pub fn byte_level() -> Self {
        let byte_by_id: Vec<u8> = (0u8..=255).collect();
        let mut id_by_byte = [0u32; 256];
        for (i, &b) in byte_by_id.iter().enumerate() {
            id_by_byte[b as usize] = i as u32;
        }
        Self {
            id_by_byte,
            byte_by_id,
            vocab_size: 256,
        }
    }

    /// Build a tokenizer restricted to the set of bytes appearing in `corpus`,
    /// so vocab_size stays small for the POC.
    pub fn from_corpus(corpus: &[u8]) -> Self {
        let mut seen: Vec<u8> = Vec::new();
        let mut present = [false; 256];
        for &b in corpus {
            if !present[b as usize] {
                present[b as usize] = true;
                seen.push(b);
            }
        }
        seen.sort_unstable();
        let byte_by_id = seen;
        let vocab_size = byte_by_id.len();
        let mut id_by_byte = [0u32; 256];
        for (i, &b) in byte_by_id.iter().enumerate() {
            id_by_byte[b as usize] = i as u32;
        }
        Self {
            id_by_byte,
            byte_by_id,
            vocab_size,
        }
    }

    /// Encode a byte slice into token ids.
    pub fn encode(&self, text: &[u8]) -> Vec<u32> {
        text.iter().map(|&b| self.id_by_byte[b as usize]).collect()
    }

    /// Decode token ids back to bytes (ignoring unknown ids).
    pub fn decode(&self, ids: &[u32]) -> Vec<u8> {
        ids.iter()
            .filter_map(|&id| self.byte_by_id.get(id as usize).copied())
            .collect()
    }
}

/// A small helper map kept for future BPE-style upgrades; unused at M0.
#[doc(hidden)]
pub type VocabMap = HashMap<u8, u32>;
