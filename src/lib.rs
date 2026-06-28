//! Core library for the `testenvironment` project.
//!
//! Logic lives here (not in `main.rs`) so it can be unit-tested directly and
//! reused by other binaries or integration tests.

/// Build a friendly greeting for the given name.
///
/// Falls back to a generic greeting when the name is empty or whitespace-only.
///
/// # Examples
///
/// ```
/// assert_eq!(testenvironment::greet("Ada"), "Hello, Ada!");
/// assert_eq!(testenvironment::greet("   "), "Hello, world!");
/// ```
pub fn greet(name: &str) -> String {
    let name = name.trim();
    if name.is_empty() {
        "Hello, world!".to_string()
    } else {
        format!("Hello, {name}!")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn greets_a_named_person() {
        assert_eq!(greet("Ada"), "Hello, Ada!");
    }

    #[test]
    fn trims_surrounding_whitespace() {
        assert_eq!(greet("  Grace  "), "Hello, Grace!");
    }

    #[test]
    fn falls_back_when_empty() {
        assert_eq!(greet(""), "Hello, world!");
        assert_eq!(greet("   "), "Hello, world!");
    }
}
