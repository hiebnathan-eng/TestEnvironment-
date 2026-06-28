//! CLI entry point.
//!
//! Usage:
//!   testenvironment [NAME]
//!
//! Prints a greeting. If NAME is omitted, greets the world.

use std::env;

use testenvironment::greet;

fn main() {
    // First positional argument is the name; everything after is ignored.
    let name = env::args().nth(1).unwrap_or_default();
    println!("{}", greet(&name));
}
