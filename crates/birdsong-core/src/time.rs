use chrono::{DateTime, Datelike, NaiveDate, Timelike, Utc};
use chrono_tz::Tz;

/// The meta model's "any week" input value.
pub const YEAR_ROUND_WEEK: i32 = -1;

/// BirdNET's 48-week year: four "weeks" per month, `1..=48`.
///
/// Days 1–7 are week 1 of the month, 8–14 week 2, 15–21 week 3, 22–31 week 4.
/// This is the scheme the location model was trained on (see `docs/DECISIONS.md` #7).
pub fn week_of_year(date: NaiveDate) -> u32 {
    (date.month() - 1) * 4 + ((date.day() - 1) / 7 + 1).min(4)
}

/// The station-local calendar date and hour (`0..=23`) of a UTC instant.
pub fn local_date_and_hour(at: DateTime<Utc>, tz: Tz) -> (NaiveDate, u32) {
    let local = at.with_timezone(&tz);
    (local.date_naive(), local.hour())
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;

    #[test]
    fn week_table() {
        let cases = [
            ((2026, 1, 1), 1),
            ((2026, 1, 7), 1),
            ((2026, 1, 8), 2),
            ((2026, 1, 21), 3),
            ((2026, 1, 22), 4),
            ((2026, 1, 29), 4),
            ((2026, 1, 31), 4),
            ((2026, 2, 1), 5),
            ((2026, 6, 15), 23),
            ((2026, 12, 31), 48),
        ];
        for ((y, m, d), want) in cases {
            let date = NaiveDate::from_ymd_opt(y, m, d).unwrap();
            assert_eq!(week_of_year(date), want, "{date}");
        }
    }

    #[test]
    fn local_parts_respect_timezone() {
        let at = Utc.with_ymd_and_hms(2026, 3, 8, 3, 30, 0).unwrap(); // 03:30 UTC
        let (date, hour) = local_date_and_hour(at, chrono_tz::America::New_York);
        assert_eq!(date, NaiveDate::from_ymd_opt(2026, 3, 7).unwrap()); // still 7 March locally
        assert_eq!(hour, 22);
        let (date, hour) = local_date_and_hour(at, chrono_tz::UTC);
        assert_eq!(date, NaiveDate::from_ymd_opt(2026, 3, 8).unwrap());
        assert_eq!(hour, 3);
    }
}
