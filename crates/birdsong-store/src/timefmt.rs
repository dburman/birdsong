//! Timestamp and date text formats used in the database.

use chrono::{DateTime, NaiveDate, SecondsFormat, Utc};

use crate::StoreError;

/// RFC 3339 UTC with exactly six fractional digits (`2026-05-01T06:00:03.000000Z`).
/// Fixed width makes lexicographic order equal chronological order. Sub-microsecond precision
/// is truncated.
pub fn format_ts(at: DateTime<Utc>) -> String {
    at.to_rfc3339_opts(SecondsFormat::Micros, true)
}

pub fn parse_ts(column: &'static str, text: &str) -> Result<DateTime<Utc>, StoreError> {
    DateTime::parse_from_rfc3339(text)
        .map(|t| t.with_timezone(&Utc))
        .map_err(|_| StoreError::Corrupt {
            column,
            value: text.to_string(),
        })
}

pub fn format_date(date: NaiveDate) -> String {
    date.format("%Y-%m-%d").to_string()
}

pub fn parse_date(column: &'static str, text: &str) -> Result<NaiveDate, StoreError> {
    NaiveDate::parse_from_str(text, "%Y-%m-%d").map_err(|_| StoreError::Corrupt {
        column,
        value: text.to_string(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::{TimeDelta, TimeZone};

    #[test]
    fn fixed_width_and_ordered() {
        let t = Utc.with_ymd_and_hms(2026, 5, 1, 6, 0, 3).unwrap();
        assert_eq!(format_ts(t), "2026-05-01T06:00:03.000000Z");
        let later = t + TimeDelta::nanoseconds(20_833);
        assert_eq!(format_ts(later), "2026-05-01T06:00:03.000020Z");
        assert!(format_ts(t) < format_ts(later));
        assert_eq!(
            parse_ts("t", &format_ts(later)).unwrap(),
            t + TimeDelta::microseconds(20)
        );
        assert!(parse_ts("t", "yesterday").is_err());
        let d = NaiveDate::from_ymd_opt(2026, 3, 8).unwrap();
        assert_eq!(parse_date("d", &format_date(d)).unwrap(), d);
    }
}
