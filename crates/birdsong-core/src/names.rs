/// Make a species name safe for file and directory names.
///
/// Every run of non-alphanumeric characters becomes a single `_`; leading and trailing `_`
/// are removed. `"Black-capped Chickadee"` → `"Black_capped_Chickadee"`.
pub fn sanitize_name(name: &str) -> String {
    let mut out = String::with_capacity(name.len());
    let mut pending_sep = false;
    for c in name.chars() {
        if c.is_alphanumeric() {
            if pending_sep && !out.is_empty() {
                out.push('_');
            }
            pending_sep = false;
            out.push(c);
        } else {
            pending_sep = true;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::sanitize_name;

    #[test]
    fn replaces_runs_and_trims() {
        assert_eq!(sanitize_name("Northern Cardinal"), "Northern_Cardinal");
        assert_eq!(
            sanitize_name("Black-capped Chickadee"),
            "Black_capped_Chickadee"
        );
        assert_eq!(sanitize_name("  Wren (sub.) "), "Wren_sub");
        assert_eq!(sanitize_name("Human vocal"), "Human_vocal");
        assert_eq!(sanitize_name("///"), "");
    }
}
