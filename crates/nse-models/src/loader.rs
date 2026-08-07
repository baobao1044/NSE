//! Save/load Toy LM weights + config via safetensors.
//!
//! Each weight matrix becomes one safetensors tensor (row-major `f32`), plus a
//! `__config__` JSON tensor holding the [`Config`]. Loading is by name, so the
//! format is resilient to small layout changes.

use std::path::Path;

use anyhow::{anyhow, Result};
use safetensors::tensor::{Dtype, SafeTensors, TensorView};

use nse_core::tensor::Matrix;
use crate::{Config, ToyLm, ToyLmWeights};

const CONFIG_TENSOR: &str = "__config__";

/// Serialize the Toy LM to a `.safetensors` file at `path`.
pub fn save_toy_lm(path: impl AsRef<Path>, lm: &ToyLm) -> Result<()> {
    // Collect (name, dtype, shape, raw_bytes) first so all byte buffers live
    // in one place and we can borrow them for `TensorView`.
    let mut entries: Vec<(String, Dtype, Vec<usize>, Vec<u8>)> = Vec::new();

    let cfg_json = serde_json::to_vec(&lm.config)?;
    entries.push((CONFIG_TENSOR.into(), Dtype::U8, vec![cfg_json.len()], cfg_json));

    push_matrix(&mut entries, "token_embed", &lm.weights.token_embed);
    for l in 0..lm.config.num_layers {
        push_matrix(&mut entries, &format!("qkv.{l}"), &lm.weights.qkv[l]);
        push_matrix(&mut entries, &format!("attn_out.{l}"), &lm.weights.attn_out[l]);
        push_matrix(&mut entries, &format!("ff_up.{l}"), &lm.weights.ff_up[l]);
        push_matrix(&mut entries, &format!("ff_down.{l}"), &lm.weights.ff_down[l]);
        push_vec(&mut entries, &format!("ln1_gain.{l}"), &lm.weights.ln1_gain[l]);
        push_vec(&mut entries, &format!("ln2_gain.{l}"), &lm.weights.ln2_gain[l]);
    }
    push_vec(&mut entries, "ln_f_gain", &lm.weights.ln_f_gain);

    // Build TensorViews borrowing from `entries`. The borrow is valid because
    // `entries` outlives `views` (both are local and dropped together).
    let views: Vec<(String, TensorView)> = entries
        .iter()
        .map(|(n, dt, sh, b)| {
            (
                n.clone(),
                TensorView::new(*dt, sh.clone(), b).expect("shape/dtype/len match"),
            )
        })
        .collect();

    let bytes = safetensors::serialize(views, &None)?;
    std::fs::write(path, &bytes)?;
    Ok(())
}

/// Load a Toy LM from a `.safetensors` file at `path`.
pub fn load_toy_lm(path: impl AsRef<Path>) -> Result<ToyLm> {
    let bytes = std::fs::read(path)?;
    let st = SafeTensors::deserialize(&bytes)?;
    let cfg = load_config(&st)?;
    let weights = load_weights(&st, &cfg)?;
    Ok(ToyLm { config: cfg, weights })
}

fn load_config(st: &SafeTensors) -> Result<Config> {
    let t = st
        .tensor(CONFIG_TENSOR)
        .map_err(|e| anyhow!("missing {CONFIG_TENSOR}: {e}"))?;
    let cfg: Config = serde_json::from_slice(t.data())?;
    Ok(cfg)
}

fn load_weights(st: &SafeTensors, cfg: &Config) -> Result<ToyLmWeights> {
    let token_embed = load_matrix(st, "token_embed", cfg.vocab_size, cfg.dim)?;
    let mut qkv = Vec::with_capacity(cfg.num_layers);
    let mut attn_out = Vec::with_capacity(cfg.num_layers);
    let mut ff_up = Vec::with_capacity(cfg.num_layers);
    let mut ff_down = Vec::with_capacity(cfg.num_layers);
    let mut ln1_gain = Vec::with_capacity(cfg.num_layers);
    let mut ln2_gain = Vec::with_capacity(cfg.num_layers);
    for l in 0..cfg.num_layers {
        qkv.push(load_matrix(st, &format!("qkv.{l}"), 3 * cfg.dim, cfg.dim)?);
        attn_out.push(load_matrix(st, &format!("attn_out.{l}"), cfg.dim, cfg.dim)?);
        ff_up.push(load_matrix(st, &format!("ff_up.{l}"), cfg.ff_dim, cfg.dim)?);
        ff_down.push(load_matrix(st, &format!("ff_down.{l}"), cfg.dim, cfg.ff_dim)?);
        ln1_gain.push(load_vec(st, &format!("ln1_gain.{l}"), cfg.dim)?);
        ln2_gain.push(load_vec(st, &format!("ln2_gain.{l}"), cfg.dim)?);
    }
    let ln_f_gain = load_vec(st, "ln_f_gain", cfg.dim)?;
    Ok(ToyLmWeights {
        token_embed,
        ln1_gain,
        qkv,
        attn_out,
        ln2_gain,
        ff_up,
        ff_down,
        ln_f_gain,
    })
}

fn load_matrix(st: &SafeTensors, name: &str, rows: usize, cols: usize) -> Result<Matrix> {
    let t = st
        .tensor(name)
        .map_err(|e| anyhow!("missing tensor {name}: {e}"))?;
    if t.dtype() != Dtype::F32 {
        return Err(anyhow!("tensor {name} expected F32, got {:?}", t.dtype()));
    }
    let data = t.data();
    let want = rows * cols * 4;
    if data.len() != want {
        return Err(anyhow!("tensor {name} size mismatch: {} != {want}", data.len()));
    }
    let mut out = vec![0.0f32; rows * cols];
    for (i, chunk) in data.chunks_exact(4).enumerate() {
        out[i] = f32::from_le_bytes(chunk.try_into().unwrap());
    }
    Ok(Matrix { rows, cols, data: out })
}

fn load_vec(st: &SafeTensors, name: &str, len: usize) -> Result<Vec<f32>> {
    let t = st
        .tensor(name)
        .map_err(|e| anyhow!("missing tensor {name}: {e}"))?;
    if t.dtype() != Dtype::F32 {
        return Err(anyhow!("tensor {name} expected F32, got {:?}", t.dtype()));
    }
    let data = t.data();
    if data.len() != len * 4 {
        return Err(anyhow!(
            "tensor {name} size mismatch: {} != {}",
            data.len(),
            len * 4
        ));
    }
    let mut out = vec![0.0f32; len];
    for (i, chunk) in data.chunks_exact(4).enumerate() {
        out[i] = f32::from_le_bytes(chunk.try_into().unwrap());
    }
    Ok(out)
}

fn push_matrix(
    out: &mut Vec<(String, Dtype, Vec<usize>, Vec<u8>)>,
    name: &str,
    m: &Matrix,
) {
    let mut bytes = Vec::with_capacity(m.data.len() * 4);
    for &v in &m.data {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    out.push((name.into(), Dtype::F32, vec![m.rows, m.cols], bytes));
}

fn push_vec(
    out: &mut Vec<(String, Dtype, Vec<usize>, Vec<u8>)>,
    name: &str,
    v: &[f32],
) {
    let mut bytes = Vec::with_capacity(v.len() * 4);
    for &x in v {
        bytes.extend_from_slice(&x.to_le_bytes());
    }
    out.push((name.into(), Dtype::F32, vec![v.len()], bytes));
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn save_load_roundtrip() {
        let cfg = Config {
            vocab_size: 16,
            dim: 8,
            num_layers: 2,
            num_heads: 2,
            max_seq_len: 32,
            ff_dim: 16,
        };
        let lm = ToyLm::init_random(cfg.clone(), 12345);
        let before = lm.forward(&[0u32, 1, 2, 3]);

        let dir = tempdir().unwrap();
        let path = dir.path().join("toy.safetensors");
        save_toy_lm(&path, &lm).unwrap();
        let lm2 = load_toy_lm(&path).unwrap();

        assert_eq!(lm2.config, cfg);
        let after = lm2.forward(&[0u32, 1, 2, 3]);
        assert_eq!(before, after);
    }
}
