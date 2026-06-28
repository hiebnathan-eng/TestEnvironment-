# testenvironment

A starter Rust CLI project.

## Requirements

- [Rust toolchain](https://rustup.rs/) (stable). Built and tested against Rust 1.94.

## Build

```sh
cargo build            # debug build
cargo build --release  # optimized build
```

## Run

```sh
cargo run -- Ada       # prints: Hello, Ada!
cargo run              # prints: Hello, world!
```

## Test

```sh
cargo test             # run the full test suite
cargo test greet       # run tests matching a name
```

## Lint & format

```sh
cargo fmt              # auto-format
cargo fmt --check      # verify formatting (CI-friendly)
cargo clippy           # lints
```

## Layout

- `src/lib.rs` — core library logic, with unit tests. Put reusable code here.
- `src/main.rs` — thin CLI wrapper around the library.
