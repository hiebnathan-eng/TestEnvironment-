# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

`testenvironment` is a Rust CLI application. It prints a greeting for a name
passed on the command line, falling back to a generic greeting when none is
given. It's a small starter scaffold — the structure (library + thin binary +
tests) is meant to be extended.

- **Language:** Rust (edition 2024), built/tested against Rust 1.94.
- **No third-party dependencies** yet — std only.

## Setup

Install the [Rust toolchain](https://rustup.rs/) (stable). No other setup or
environment variables are required.

## Common commands

- **Build:** `cargo build` (debug) / `cargo build --release` (optimized)
- **Run:** `cargo run -- <NAME>` (e.g. `cargo run -- Ada`); `cargo run` greets the world
- **Test (all):** `cargo test` — runs unit tests and doctests
- **Test (single):** `cargo test <substring>` (e.g. `cargo test greet`)
- **Format:** `cargo fmt` (apply) / `cargo fmt --check` (verify, CI-friendly)
- **Lint:** `cargo clippy`

## Architecture

The crate ships both a library and a binary that share a name (`testenvironment`):

- `src/lib.rs` — **core logic lives here.** Public functions are unit-tested in
  an inline `#[cfg(test)] mod tests` block, and documented with doctests that
  also run under `cargo test`. Add reusable code here, not in `main.rs`.
- `src/main.rs` — a thin CLI wrapper: it parses arguments and delegates to the
  library. Keep it minimal so the logic stays testable.

This split is deliberate — keeping logic in the library means it can be tested
directly and reused, while `main.rs` only handles I/O and argument wiring.

## Conventions

- Keep business logic in `src/lib.rs`; `main.rs` should only do I/O/arg parsing.
- Add unit tests alongside new library code in the `tests` module, and a
  doctest example on public functions where it aids understanding.
- Before committing, run `cargo fmt`, `cargo clippy`, and `cargo test`.

## Working agreements

- Keep this file accurate. When you change how the project is built, run, or
  tested, update the relevant section here in the same change.
- Prefer documenting the *why* and the non-obvious over restating what is
  already clear from reading the code.
