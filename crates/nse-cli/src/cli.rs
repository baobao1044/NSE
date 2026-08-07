//! CLI dispatch. Skeleton (M0): parses subcommands but each handler is a stub
//! that prints a "not yet implemented" message. Wiring lands in M5.

use anyhow::Result;

/// Entry point invoked by `main`.
pub fn run() -> Result<()> {
    eprintln!("nse: Neuro-Sparse Engine CLI (skeleton)");
    eprintln!("Subcommands (M5): train | eval dense | transmute | eval sparse | eval compare");
    eprintln!("No subcommand given; nothing to do. Try `nse --help` once wired up.");
    Ok(())
}
