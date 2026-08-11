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
//! nse train-composite -> toy_lm_comp.safetensors (hippocampus+cortex pipeline, M7)
//! nse eval dense     -> PPL_dense            (baseline perplexity)
//! nse transmute      -> model.nse            (ZSTM: outlier + k-means + ternary)
//! nse eval sparse     -> PPL_sparse          (RIE + LLER, --kernel/--index)
//! nse eval compare    -> report             (PPL_dense | PPL_sparse | % drop)
//! nse eval-composite  -> 4-path report      (dense/sparse × GELU/Hopfield, M7)
//! ```
//!
//! `--kernel scalar|avx2|auto` and `--index brute|hnsw` select the LLER/RIE
//! backends used by the sparse eval path (N1/N2). `--beta` sets the Hopfield
//! retrieval sharpness for the composite eval (M7).

#![allow(dead_code)]

use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};

use nse_eval::{
    compare_composite, compare_with_options, dense_ppl, sparse_ppl_with_options, Activation,
    CompositeReport, SparseOptions,
};
use nse_ller::KernelKind;
use nse_models::{Config, Tokenizer, ToyLm};
use nse_rie::IndexKind;
use nse_train::{
    CompositeConfig, CompositeTrainer, ForwardForwardConfig, ForwardForwardTrainer,
    HopfieldConfig, HopfieldTrainer, LshSparseConfig, LshSparseTrainer, SgdConfig, SgdTrainer,
    Trainer,
};
use nse_zstm::{transmute, save_transmuted, load_transmuted, QuantSchemeConfig, TransmuteConfig};

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
        /// Per-weight max-norm clamp (FF stabilization; 0 disables).
        #[arg(long, default_value_t = 1.0)]
        weight_clip: f32,
        /// Goodness normalization (FF homeostasis): "none" (raw G, needs weight_clip)
        /// or "layernorm" (standardize G — experimentally fails, kept for repro).
        #[arg(long, default_value = "none")]
        homeostasis: String,
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
        /// Scale of the value written into the FFN (after unit-normalizing).
        #[arg(long, default_value_t = 0.1)]
        value_scale: f32,
    },
    /// Train the Toy LM with LSH-sparse updates (dense backprop + per-row LSH
    /// gradient masking). Closest to SGD; updates ~sparse_fraction of rows/step.
    TrainLsh {
        #[arg(long, default_value = "data/corpus.txt")]
        corpus: PathBuf,
        #[arg(long, default_value = "toy_lm_lsh.safetensors")]
        out: PathBuf,
        /// Optional warm-start: load this model instead of random init (e.g. a
        /// Forward-Forward-trained model) then fine-tune with LSH-sparse updates.
        #[arg(long)]
        init: Option<PathBuf>,
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
    /// Train the Toy LM with the composite "hippocampus + cortex" pipeline:
    /// SGD warm → Hopfield writes → Forward-Forward (local) → LSH-sparse
    /// fine-tune. Each phase is skipped when its epoch/write count is 0.
    /// Research prototype; bar = beat or match SGD at comparable compute.
    TrainComposite {
        #[arg(long, default_value = "data/corpus.txt")]
        corpus: PathBuf,
        #[arg(long, default_value = "toy_lm_comp.safetensors")]
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
        /// Phase 1: SGD warm epochs (0 skips — off by default, FF warm is the
        /// stabilizer per paper §5.4.2).
        #[arg(long, default_value_t = 0)]
        sgd_epochs: usize,
        /// Phase 2: Hopfield one-shot writes per layer (0 skips — off by
        /// default due to the dense-PPL mismatch, §5.4.3).
        #[arg(long, default_value_t = 0)]
        hopfield_writes: usize,
        /// Phase 3: Forward-Forward epochs (warm-start, 0 skips).
        #[arg(long, default_value_t = 15)]
        ff_epochs: usize,
        /// Phase 4: LSH-sparse fine-tune epochs (0 skips).
        #[arg(long, default_value_t = 15)]
        lsh_epochs: usize,
        /// FF max-norm clamp (homeostasis sweet spot 0.5, paper §5.4).
        #[arg(long, default_value_t = 0.5)]
        ff_clip: f32,
        /// LSH sparse-fraction (fraction of rows updated/step).
        #[arg(long, default_value_t = 0.01)]
        lsh_frac: f32,
        /// Retrieval sharpness β used by the between-phase PPL probe.
        #[arg(long, default_value_t = 8.0)]
        eval_beta: f32,
    },
    /// Evaluate dense PPL of a trained model.
    EvalDense {
        #[arg(long, default_value = "data/corpus.txt")]
        corpus: PathBuf,
        #[arg(long, default_value = "toy_lm.safetensors")]
        model: PathBuf,
        #[arg(long, default_value_t = 16)]
        seq_len: usize,
        /// FFN forward path: "gelu" (standard) or "hopfield" (softmax retrieval).
        /// Tests whether a Hopfield-trained model does better under the
        /// retrieval path it was designed for (architecture-mismatch test).
        #[arg(long, default_value = "gelu")]
        forward: String,
        /// Retrieval sharpness β (only used with --forward hopfield).
        #[arg(long, default_value_t = 8.0)]
        beta: f32,
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
        /// Quantization scheme for the expert weights:
        /// "ternary" (default, `{-1,0,1}` + per-row scale) or "pq" (Product
        /// Quantization, 8-bit shared codebook per layer — Phase 7 / M8).
        #[arg(long, default_value = "ternary")]
        quant: String,
        /// Number of PQ sub-vectors `M` (only used with `--quant pq`).
        /// `in_dim` must be divisible by `M`; the largest divisor `<= M` is
        /// used if not. Default 4 → `sub_dim = in_dim/4` (e.g. 16 for dim=64).
        #[arg(long, default_value_t = 4)]
        pq_subvectors: usize,
        /// PQ codebook bits per code (only used with `--quant pq`).
        /// `8` → 256 centroids per sub-codebook (plan default).
        #[arg(long, default_value_t = 8)]
        pq_nbits: usize,
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
    /// Compare all four forward paths (dense/sparse × GELU/Hopfield) and print
    /// the composite report — the §5.6 artifact. `beta` is the Hopfield
    /// retrieval sharpness (used for both dense and sparse retrieval paths).
    EvalComposite {
        #[arg(long, default_value = "data/corpus.txt")]
        corpus: PathBuf,
        #[arg(long, default_value = "toy_lm_comp.safetensors")]
        model: PathBuf,
        #[arg(long, default_value = "model.nse")]
        nse: PathBuf,
        #[arg(long, default_value_t = 16)]
        seq_len: usize,
        /// Hopfield retrieval sharpness β.
        #[arg(long, default_value_t = 8.0)]
        beta: f32,
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
        Cmd::TrainFf { corpus, out, dim, layers, heads, seq_len, ff_dim, epochs, lr, hebb_lr, weight_clip, homeostasis } => {
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
            let homeo = match homeostasis.to_ascii_lowercase().as_str() {
                "none" => nse_train::Homeostasis::None,
                "layernorm" => nse_train::Homeostasis::LayerNorm,
                _ => anyhow::bail!("invalid --homeostasis '{homeostasis}': expected none|layernorm"),
            };
            eprintln!("Training Toy LM (Forward-Forward): homeostasis={homeostasis}, weight_clip={weight_clip}");
            let mut trainer = ForwardForwardTrainer::new(ForwardForwardConfig {
                learning_rate: lr,
                seq_len,
                epochs,
                hebbian_embed_lr: hebb_lr,
                weight_clip,
                homeostasis: homeo,
                log_every: 10,
                ..Default::default()
            });
            trainer.train(&mut lm, &corpus_bytes)?;
            nse_models::loader::save_toy_lm(&out, &lm)?;
            eprintln!("Saved FF-trained model to {}", out.display());
            Ok(())
        }
        Cmd::TrainHopfield { corpus, out, dim, layers, heads, seq_len, ff_dim, num_writes, beta, value_scale } => {
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
            eprintln!("Training Toy LM (Hopfield): one-shot associative writes into FFN, value_scale={value_scale}");
            let mut trainer = HopfieldTrainer::new(HopfieldConfig {
                seq_len,
                num_writes,
                beta,
                value_scale,
                log_every: 1,
                ..Default::default()
            });
            trainer.train(&mut lm, &corpus_bytes)?;
            nse_models::loader::save_toy_lm(&out, &lm)?;
            eprintln!("Saved Hopfield-trained model to {}", out.display());
            Ok(())
        }
        Cmd::TrainLsh { corpus, out, init, dim, layers, heads, seq_len, ff_dim, epochs, lr, sparse_fraction } => {
            let corpus_bytes = std::fs::read(&corpus)
                .with_context(|| format!("reading corpus {}", corpus.display()))?;
            let tok = Tokenizer::from_corpus(&corpus_bytes);
            let mut lm = if let Some(p) = &init {
                eprintln!("LSH-sparse: warm-starting from {}", p.display());
                nse_models::loader::load_toy_lm(p)?
            } else {
                let cfg = Config {
                    vocab_size: tok.vocab_size,
                    dim,
                    num_layers: layers,
                    num_heads: heads,
                    max_seq_len: seq_len,
                    ff_dim,
                };
                ToyLm::init_random(cfg, 1337)
            };
            eprintln!("Training Toy LM (LSH-sparse): dense backprop + per-row LSH grad mask{}", if init.is_some() { " [warm-start]" } else { "" });
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
        Cmd::TrainComposite {
            corpus, out, dim, layers, heads, seq_len, ff_dim,
            sgd_epochs, hopfield_writes, ff_epochs, lsh_epochs, ff_clip, lsh_frac, eval_beta,
        } => {
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
            eprintln!(
                "Training Toy LM (composite): SGD {sgd_epochs} ep + Hopfield {hopfield_writes} writes + FF {ff_epochs} ep (clip {ff_clip}) + LSH {lsh_epochs} ep (frac {lsh_frac})"
            );
            let sgd_cfg = SgdConfig {
                learning_rate: 0.05, seq_len, epochs: sgd_epochs, lr_decay: 1.0,
                log_every: 0, seed: 1337,
            };
            let hop_cfg = HopfieldConfig {
                seq_len, num_writes: hopfield_writes, beta: eval_beta, value_scale: 0.1,
                log_every: 0, seed: 3,
            };
            let ff_cfg = ForwardForwardConfig {
                learning_rate: 0.02, seq_len, epochs: ff_epochs, hebbian_embed_lr: 0.01,
                weight_clip: ff_clip, log_every: 10, ..Default::default()
            };
            let lsh_cfg = LshSparseConfig {
                learning_rate: 0.05, seq_len, epochs: lsh_epochs,
                sparse_fraction: lsh_frac, log_every: 10, ..Default::default()
            };
            let mut trainer = CompositeTrainer::new(CompositeConfig {
                sgd_warm: sgd_cfg,
                hopfield: hop_cfg,
                ff: ff_cfg,
                lsh: lsh_cfg,
                eval_seq_len: seq_len,
                eval_beta,
                log_every: 1,
            });
            trainer.train(&mut lm, &corpus_bytes)?;
            nse_models::loader::save_toy_lm(&out, &lm)?;
            eprintln!("Saved composite-trained model to {}", out.display());
            Ok(())
        }
        Cmd::EvalDense { corpus, model, seq_len, forward, beta } => {
            let corpus_bytes = std::fs::read(&corpus)?;
            let lm = nse_models::loader::load_toy_lm(&model)?;
            let tok = Tokenizer::from_corpus(&corpus_bytes);
            let ids = tok.encode(&corpus_bytes);
            let ppl = match forward.to_ascii_lowercase().as_str() {
                "gelu" => dense_ppl(&lm, &ids, seq_len),
                "hopfield" => nse_eval::dense_ppl_hopfield(&lm, &ids, seq_len, beta),
                _ => anyhow::bail!("invalid --forward '{forward}': expected gelu|hopfield"),
            };
            println!("PPL (dense, forward={forward}): {:.4}", ppl);
            Ok(())
        }
        Cmd::Transmute { corpus, model, out, outlier_fraction, quant, pq_subvectors, pq_nbits } => {
            let corpus_bytes = std::fs::read(&corpus)?;
            let lm = nse_models::loader::load_toy_lm(&model)?;
            let quant_scheme = match quant.as_str() {
                "ternary" => {
                    eprintln!("Quantization scheme: ternary ({{-1,0,1}} + per-row scale)");
                    QuantSchemeConfig::Ternary
                }
                "pq" => {
                    eprintln!(
                        "Quantization scheme: PQ (M={pq_subvectors} sub-vectors, {pq_nbits}-bit codebook)"
                    );
                    QuantSchemeConfig::Pq {
                        num_sub_vectors: pq_subvectors,
                        nbits: pq_nbits,
                        iters: 20,
                        seed: 7,
                    }
                }
                other => {
                    anyhow::bail!("unknown --quant value '{other}': expected 'ternary' or 'pq'");
                }
            };
            let cfg = TransmuteConfig {
                outlier: nse_zstm::outlier::OutlierConfig { fraction: outlier_fraction },
                cluster: nse_zstm::cluster::ClusterConfig {
                    num_experts: 0,
                    iters: 10,
                    seed: 7,
                },
                quant: quant_scheme,
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
        Cmd::EvalComposite { corpus, model, nse, seq_len, beta, kernel, index } => {
            let corpus_bytes = std::fs::read(&corpus)?;
            let lm = nse_models::loader::load_toy_lm(&model)?;
            let tm = load_transmuted(&nse)?;
            let opts = SparseOptions {
                kernel: parse_kernel(&kernel)?,
                index: parse_index(&index)?,
            };
            let report: CompositeReport = compare_composite(
                &lm, &tm, &corpus_bytes, seq_len, beta, Activation::All, opts,
            );
            println!("{}", report.pretty());
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
