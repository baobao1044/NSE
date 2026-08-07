//! `nse` — Neuro-Sparse Engine command-line interface.
//!
//! Drives the POC pipeline as independent subcommands, each producing an
//! intermediate artifact so every stage can be debugged separately:
//!
//! ```text
//! nse train          -> toy_lm.safetensors   (train Toy LM, SgdTrainer)
//! nse train-ff       -> toy_lm.safetensors   (Forward-Forward trainer)
//! nse train-hopfield -> toy_lm.safetensors   (Hopfield associative writes)
//! nse train-lsh      -> toy_lm.safetensors   (LSH-sparse trainer)
//! nse eval dense     -> PPL_dense            (baseline perplexity)
//! nse transmute      -> model.nse            (ZSTM: outlier + k-means + ternary)
//! nse eval sparse     -> PPL_sparse          (RIE + LLER, --kernel/--index)
//! nse eval compare    -> report             (PPL_dense | PPL_sparse | % drop)
//! ```
//!
//! `--kernel scalar|avx2|auto` and `--index brute|hnsw` select the LLER/RIE
//! backends used by the sparse eval path (N1/N2).

#![allow(dead_code)]

use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};

use nse_eval::{compare_with_options, dense_ppl, sparse_ppl_with_options, Activation, SparseOptions};
use nse_ller::KernelKind;
use nse_models::{Config, Tokenizer, ToyLm};
use nse_rie::IndexKind;
use nse_train::{ForwardForwardConfig, ForwardForwardTrainer, HopfieldConfig, HopfieldTrainer, LshSparseConfig, LshSparseTrainer, SgdConfig, SgdTrainer, Trainer};
use nse_zstm::{transmute, save_transmuted, load_transmuted, TransmuteConfig};

#[derive(Parser, Debug)]
#[command(name = "nse", about = "Neuro-Sparse Engine POC CLI")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand, Debug)]
enum Cmd {
    /// Train the Toy LM (SGD baseline) and save to a safetensors file.
    Train {
        #[arg(long, default_value = "data/corpus.txt")]
        corpus: PathBuf,
        #[arg(long, default_value = "toy_lm.safetensors")]
        out: PathBuf,
        #[arg(long, default_value_t = 32)]
        dim: usize,
        #[arg(long, default_value_t = 2)]
        layers: usize,
        #[arg(long, default_value_t = 4)]
        heads: usize,
        #[arg(long, default_value_t = 32)]
        seq_len: usize,
        #[arg(long, default_value_t = 64)]
        ff_dim: usize,
        #[arg(long, default_value_t = 80)]
        epochs: usize,
        #[arg(long, default_value_t = 0.05)]
        lr: f32,
    },
    /// Train the Toy LM with the Forward-Forward algorithm (local goodness,
    /// no global backprop) + light Hebbian head. Research prototype.
    TrainFf {
        #[arg(long, default_value = "data/corpus.txt")]
        corpus: PathBuf,
        #[arg(long, default_value = "toy_lm_ff.safetensors")]
        out: PathBuf,
        #[arg(long, default_value_t = 32)]
        dim: usize,
        #[arg(long, default_value_t = 2)]
        layers: usize,
        #[arg(long, default_value_t = 4)]
        heads: usize,
        #[arg(long, default_value_t = 16)]
        seq_len: usize,
        #[arg(long, default_value_t = 64)]
        ff_dim: usize,
        #[arg(long, default_value_t = 60)]
        epochs: usize,
        #[arg(long, default_value_t = 0.02)]
        lr: f32,
        #[arg(long, default_value_t = 0.01)]
        hebb_lr: f32,
    },
    /// Train the Toy LM by writing associative memories into the FFN
    /// (modern Hopfield, one-shot writes, no backprop). Research prototype.
    TrainHopfield {
        #[arg(long, default_value = "data/corpus.txt")]
        corpus: PathBuf,
        #[arg(long, default_value = "toy_lm_hop.safetensors")]
        out: PathBuf,
        #[arg(long, default_value_t = 32)]
        dim: usize,
        #[arg(long, default_value_t = 2)]
        layers: usize,
        #[arg(long, default_value_t = 4)]
        heads: usize,
        #[arg(long, default_value_t = 16)]
        seq_len: usize,
        #[arg(long, default_value_t = 64)]
        ff_dim: usize,
        #[arg(long, default_value_t = 64)]
        num_writes: usize,
        #[arg(long, default_value_t = 8.0)]
        beta: f32,
    },
    /// Train the Toy LM with LSH-sparse updates (dense backprop + per-row LSH
    /// gradient masking). Closest to SGD; updates ~sparse_fraction of rows/step.
    TrainLsh {
        #[arg(long, default_value = "data/corpus.txt")]
        corpus: PathBuf,
        #[arg(long, default_value = "toy_lm_lsh.safetensors")]
        out: PathBuf,
        #[arg(long, default_value_t = 32)]
        dim: usize,
        #[arg(long, default_value_t = 2)]
        layers: usize,
        #[arg(long, default_value_t = 4)]
        heads: usize,
        #[arg(long, default_value_t = 16)]
        seq_len: usize,
        #[arg(long, default_value_t = 64)]
        ff_dim: usize,
        #[arg(long, default_value_t = 40)]
        epochs: usize,
        #[arg(long, default_value_t = 0.05)]
        lr: f32,
        #[arg(long, default_value_t = 0.01)]
        sparse_fraction: f32,
    },
    /// Evaluate dense PPL of a trained model.
    EvalDense {
        #[arg(long, default_value = "data/corpus.txt")]
        corpus: PathBuf,
        #[arg(long, default_value = "toy_lm.safetensors")]
        model: PathBuf,
        #[arg(long, default_value_t = 16)]
        seq_len: usize,
    },
    /// Transmute a trained model into the sparse NSE format (.nse JSON).
    Transmute {
        #[arg(long, default_value = "data/corpus.txt")]
        corpus: PathBuf,
        #[arg(long, default_value = "toy_lm.safetensors")]
        model: PathBuf,
        #[arg(long, default_value = "model.nse")]
        out: PathBuf,
        #[arg(long, default_value_t = 0.1)]
        outlier_fraction: f32,
    },
    /// Evaluate sparse PPL of a transmuted model.
    EvalSparse {
        #[arg(long, default_value = "data/corpus.txt")]
        corpus: PathBuf,
        #[arg(long, default_value = "model.nse")]
        nse: PathBuf,
        #[arg(long, default_value_t = 16)]
        seq_len: usize,
        /// Routing mode: "all" (all experts, upper bound) or "threshold".
        #[arg(long, default_value = "all")]
        mode: String,
        #[arg(long, default_value_t = 0.5)]
        threshold_ratio: f32,
        #[arg(long, default_value_t = 16)]
        max_k: usize,
        /// LLER kernel backend: "scalar" | "avx2" | "auto" (auto = AVX2 if CPU has it).
        #[arg(long, default_value = "auto")]
        kernel: String,
        /// RIE index backend: "brute" (exact MIPS) | "hnsw" (approximate).
        #[arg(long, default_value = "brute")]
        index: String,
    },
    /// Compare dense vs sparse PPL and print the headline report.
    EvalCompare {
        #[arg(long, default_value = "data/corpus.txt")]
        corpus: PathBuf,
        #[arg(long, default_value = "toy_lm.safetensors")]
        model: PathBuf,
        #[arg(long, default_value = "model.nse")]
        nse: PathBuf,
        #[arg(long, default_value_t = 16)]
        seq_len: usize,
        /// LLER kernel backend: "scalar" | "avx2" | "auto".
        #[arg(long, default_value = "auto")]
        kernel: String,
        /// RIE index backend: "brute" | "hnsw".
        #[arg(long, default_value = "brute")]
        index: String,
    },
}

/// Entry point invoked by `main`.
pub fn run() -> Result<()> {
    let cli = Cli::parse();
    match cli.cmd {
        Cmd::Train { corpus, out, dim, layers, heads, seq_len, ff_dim, epochs, lr } => {
            let corpus_bytes = std::fs::read(&corpus)
                .with_context(|| format!("reading corpus {}", corpus.display()))?;
            let tok = Tokenizer::from_corpus(&corpus_bytes);
            let cfg = Config {
                vocab_size: tok.vocab_size,
                dim,
                num_layers: layers,
                num_heads: heads,
                max_seq_len: seq_len,
                ff_dim,
            };
            let mut lm = ToyLm::init_random(cfg.clone(), 1337);
            eprintln!("Training Toy LM: {} vocab, {} dim, {} layers", cfg.vocab_size, cfg.dim, cfg.num_layers);
            let mut trainer = SgdTrainer::new(SgdConfig {
                learning_rate: lr,
                seq_len,
                epochs,
                lr_decay: 1.0,
                log_every: 0,
                seed: 1337,
            });
            trainer.train(&mut lm, &corpus_bytes)?;
            nse_models::loader::save_toy_lm(&out, &lm)?;
            eprintln!("Saved trained model to {}", out.display());
            Ok(())
        }
        Cmd::TrainFf { corpus, out, dim, layers, heads, seq_len, ff_dim, epochs, lr, hebb_lr } => {
            let corpus_bytes = std::fs::read(&corpus)
                .with_context(|| format!("reading corpus {}", corpus.display()))?;
            let tok = Tokenizer::from_corpus(&corpus_bytes);
            let cfg = Config {
                vocab_size: tok.vocab_size,
                dim,
                num_layers: layers,
                num_heads: heads,
                max_seq_len: seq_len,
                ff_dim,
            };
            let mut lm = ToyLm::init_random(cfg.clone(), 1337);
            eprintln!("Training Toy LM (Forward-Forward): local goodness, no global backprop");
            let mut trainer = ForwardForwardTrainer::new(ForwardForwardConfig {
                learning_rate: lr,
                seq_len,
                epochs,
                hebbian_embed_lr: hebb_lr,
                log_every: 10,
                ..Default::default()
            });
            trainer.train(&mut lm, &corpus_bytes)?;
            nse_models::loader::save_toy_lm(&out, &lm)?;
            eprintln!("Saved FF-trained model to {}", out.display());
            Ok(())
        }
        Cmd::TrainHopfield { corpus, out, dim, layers, heads, seq_len, ff_dim, num_writes, beta } => {
            let corpus_bytes = std::fs::read(&corpus)
                .with_context(|| format!("reading corpus {}", corpus.display()))?;
            let tok = Tokenizer::from_corpus(&corpus_bytes);
            let cfg = Config {
                vocab_size: tok.vocab_size,
                dim,
                num_layers: layers,
                num_heads: heads,
                max_seq_len: seq_len,
                ff_dim,
            };
            let mut lm = ToyLm::init_random(cfg.clone(), 1337);
            eprintln!("Training Toy LM (Hopfield): one-shot associative writes into FFN");
            let mut trainer = HopfieldTrainer::new(HopfieldConfig {
                seq_len,
                num_writes,
                beta,
                log_every: 1,
                ..Default::default()
            });
            trainer.train(&mut lm, &corpus_bytes)?;
            nse_models::loader::save_toy_lm(&out, &lm)?;
            eprintln!("Saved Hopfield-trained model to {}", out.display());
            Ok(())
        }
        Cmd::TrainLsh { corpus, out, dim, layers, heads, seq_len, ff_dim, epochs, lr, sparse_fraction } => {
            let corpus_bytes = std::fs::read(&corpus)
                .with_context(|| format!("reading corpus {}", corpus.display()))?;
            let tok = Tokenizer::from_corpus(&corpus_bytes);
            let cfg = Config {
                vocab_size: tok.vocab_size,
                dim,
                num_layers: layers,
                num_heads: heads,
                max_seq_len: seq_len,
                ff_dim,
            };
            let mut lm = ToyLm::init_random(cfg.clone(), 1337);
            eprintln!("Training Toy LM (LSH-sparse): dense backprop + per-row LSH grad mask");
            let mut trainer = LshSparseTrainer::new(LshSparseConfig {
                learning_rate: lr,
                seq_len,
                epochs,
                sparse_fraction,
                log_every: 10,
                ..Default::default()
            });
            trainer.train(&mut lm, &corpus_bytes)?;
            nse_models::loader::save_toy_lm(&out, &lm)?;
            eprintln!("Saved LSH-trained model to {}", out.display());
            Ok(())
        }
        Cmd::EvalDense { corpus, model, seq_len } => {
            let corpus_bytes = std::fs::read(&corpus)?;
            let lm = nse_models::loader::load_toy_lm(&model)?;
            let tok = Tokenizer::from_corpus(&corpus_bytes);
            let ids = tok.encode(&corpus_bytes);
            let ppl = dense_ppl(&lm, &ids, seq_len);
            println!("PPL (dense): {:.4}", ppl);
            Ok(())
        }
        Cmd::Transmute { corpus, model, out, outlier_fraction } => {
            let corpus_bytes = std::fs::read(&corpus)?;
            let lm = nse_models::loader::load_toy_lm(&model)?;
            let cfg = TransmuteConfig {
                outlier: nse_zstm::outlier::OutlierConfig { fraction: outlier_fraction },
                ..TransmuteConfig::poc()
            };
            eprintln!("Transmuting dense model -> sparse NSE format");
            let tm = transmute(&lm, Some(&corpus_bytes), &cfg)?;
            save_transmuted(&tm, &out)?;
            eprintln!("Saved transmuted model to {}", out.display());
            Ok(())
        }
        Cmd::EvalSparse { corpus, nse, seq_len, mode, threshold_ratio, max_k, kernel, index } => {
            let corpus_bytes = std::fs::read(&corpus)?;
            let tm = load_transmuted(&nse)?;
            let tok = Tokenizer::from_corpus(&corpus_bytes);
            let ids = tok.encode(&corpus_bytes);
            let act = if mode == "threshold" {
                Activation::Threshold { ratio: threshold_ratio, max_k }
            } else {
                Activation::All
            };
            let opts = SparseOptions {
                kernel: parse_kernel(&kernel)?,
                index: parse_index(&index)?,
            };
            let ppl = sparse_ppl_with_options(&tm, &ids, seq_len, act, opts);
            println!("PPL (sparse, mode={mode}, kernel={kernel}, index={index}): {:.4}", ppl);
            Ok(())
        }
        Cmd::EvalCompare { corpus, model, nse, seq_len, kernel, index } => {
            let corpus_bytes = std::fs::read(&corpus)?;
            let lm = nse_models::loader::load_toy_lm(&model)?;
            let tm = load_transmuted(&nse)?;
            let opts = SparseOptions {
                kernel: parse_kernel(&kernel)?,
                index: parse_index(&index)?,
            };
            let report = compare_with_options(&lm, &tm, &corpus_bytes, seq_len, Activation::All, opts);
            println!("{}\n{}", "=== NSE POC: Dense vs Sparse PPL ===", report.pretty());
            Ok(())
        }
    }
}

/// Parse the `--kernel` flag. Accepts "scalar" | "avx2" | "auto".
fn parse_kernel(s: &str) -> Result<KernelKind> {
    match s.to_ascii_lowercase().as_str() {
        "scalar" => Ok(KernelKind::Scalar),
        "avx2" => Ok(KernelKind::Avx2),
        "auto" => Ok(KernelKind::Auto),
        _ => anyhow::bail!("invalid --kernel '{s}': expected scalar|avx2|auto"),
    }
}

/// Parse the `--index` flag. Accepts "brute" | "hnsw".
fn parse_index(s: &str) -> Result<IndexKind> {
    match s.to_ascii_lowercase().as_str() {
        "brute" => Ok(IndexKind::Brute),
        "hnsw" => Ok(IndexKind::Hnsw),
        _ => anyhow::bail!("invalid --index '{s}': expected brute|hnsw"),
    }
}
