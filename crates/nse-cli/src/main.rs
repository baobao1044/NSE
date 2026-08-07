//! `nse` — Neuro-Sparse Engine command-line interface.
//!
//! Drives the POC pipeline as independent subcommands, each producing an
//! intermediate artifact so every stage can be debugged separately:
//!
//! ```text
//! nse train      -> toy_lm.safetensors   (train Toy LM, SgdTrainer)
//! nse eval dense -> PPL_dense            (baseline perplexity)
//! nse transmute  -> model.nse            (ZSTM: outlier + k-means + ternary)
//! nse eval sparse -> PPL_sparse          (RIE + LLER scalar + bias)
//! nse eval compare -> report             (PPL_dense | PPL_sparse | % drop)
//! ```

mod cli;

fn main() -> anyhow::Result<()> {
    cli::run()
}
