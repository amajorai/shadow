/// Strip markdown fences and extract the first complete JSON object from `text`.
pub(crate) fn extract_json(text: &str) -> Option<String> {
    let stripped = text
        .trim()
        .trim_start_matches("```json")
        .trim_start_matches("```")
        .trim_end_matches("```")
        .trim();
    let start = stripped.find('{')?;
    let end = stripped.rfind('}')?;
    if end >= start {
        Some(stripped[start..=end].to_string())
    } else {
        None
    }
}

pub(crate) fn wall_micros() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_micros() as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extract_json_from_bare_object() {
        assert_eq!(extract_json(r#"{"a":1}"#), Some(r#"{"a":1}"#.to_string()));
    }

    #[test]
    fn extract_json_strips_json_fence() {
        let text = "```json\n{\"a\":1,\"b\":2}\n```";
        assert_eq!(extract_json(text), Some(r#"{"a":1,"b":2}"#.to_string()));
    }

    #[test]
    fn extract_json_strips_plain_fence() {
        let text = "```\n{\"ok\":true}\n```";
        assert_eq!(extract_json(text), Some(r#"{"ok":true}"#.to_string()));
    }

    #[test]
    fn extract_json_ignores_prose_around_object() {
        let text = "Sure! Here is the result:\n{\"x\": 42}\nHope that helps.";
        assert_eq!(extract_json(text), Some(r#"{"x": 42}"#.to_string()));
    }

    #[test]
    fn extract_json_spans_from_first_brace_to_last_brace() {
        // Nested braces: must capture the outermost span, not stop at the first close.
        let text = r#"prefix {"outer": {"inner": 1}} suffix"#;
        assert_eq!(
            extract_json(text),
            Some(r#"{"outer": {"inner": 1}}"#.to_string())
        );
    }

    #[test]
    fn extract_json_none_when_no_braces() {
        assert_eq!(extract_json("no json here"), None);
        assert_eq!(extract_json(""), None);
    }

    #[test]
    fn extract_json_none_when_only_open_brace() {
        // rfind('}') fails → None.
        assert_eq!(extract_json("{ incomplete"), None);
    }

    #[test]
    fn extract_json_captures_inclusive_closing_brace() {
        // The returned slice is inclusive of the final `}` (start..=end).
        let out = extract_json("  {\"k\":\"v\"}  ").unwrap();
        assert!(out.starts_with('{') && out.ends_with('}'));
    }

    #[test]
    fn wall_micros_is_monotonic_nondecreasing_and_plausible() {
        let a = wall_micros();
        let b = wall_micros();
        assert!(b >= a, "wall clock went backwards: {a} then {b}");
        // Sanity: well past 2020 (1.5e15 microseconds since epoch ~= 2017).
        assert!(a > 1_500_000_000_000_000, "implausibly small: {a}");
    }
}
