//! Datetime encoding used by journal logs.
//!
//! `PersistedTrial` carries timezone-naive UTC (see `rustuna_core::datetime`), which
//! [`crate::sqlite3::SQLite3Storage`] stores as-is. Journal logs instead carry timezone-aware UTC,
//! matching Optuna, because a log entry is JSON and can spell out the offset without the schema
//! concerns a database column has.

use rustuna_core::internal::datetime::now_naive_utc;

/// Width of the `YYYY-MM-DDTHH:MM:SS` prefix every journal datetime starts with.
const DATE_TIME_LEN: usize = 19;

/// Returns the current time in the timezone-aware UTC form journal logs carry.
///
/// The output matches Optuna's `datetime.now(tz=timezone.utc).isoformat(timespec="microseconds")`.
pub fn now_aware_utc() -> String {
    naive_utc_to_aware_utc(&now_naive_utc())
}

/// Rewrites a timezone-naive UTC timestamp as the timezone-aware form used by journal logs.
pub fn naive_utc_to_aware_utc(value: &str) -> String {
    format!("{}+00:00", value.replacen(' ', "T", 1))
}

/// Converts a datetime read from a journal log into the timezone-naive UTC that `PersistedTrial`
/// holds.
///
/// **The offset is dropped, not applied.** Optuna and Rustuna both write `+00:00`, so there is
/// nothing to apply, and skipping it keeps this crate free of date arithmetic. The cost is that a
/// log written by some other tool with a real offset would be read as though its wall-clock reading
/// were already UTC. Accepting such a value unchanged is still better than rejecting it, since the
/// datetimes are informational.
///
/// Logs predating Optuna 5.0.0rc1 carry a naive datetime with no offset at all. Those were local
/// time and are likewise taken as UTC.
pub fn journal_datetime_to_naive_utc(value: &str) -> String {
    let bytes = value.as_bytes();
    // Leave anything that is not shaped like a datetime exactly as it was found.
    if bytes.len() < DATE_TIME_LEN
        || bytes[4] != b'-'
        || bytes[7] != b'-'
        // RFC 3339 section 5.6 allows any of these to separate the date from the time.
        || !matches!(bytes[10], b'T' | b't' | b' ')
        || bytes[13] != b':'
        || bytes[16] != b':'
    {
        return value.to_string();
    }

    // Whatever follows the seconds is the fractional part plus the offset. Keep the digits of the
    // former, at microsecond precision, and discard the latter.
    let trailer = &value[DATE_TIME_LEN..];
    let fraction = trailer
        .strip_prefix('.')
        .map(|digits| {
            let end = digits
                .find(|c: char| !c.is_ascii_digit())
                .unwrap_or(digits.len());
            &digits[..end.min(6)]
        })
        .unwrap_or("");
    format!(
        "{} {}.{fraction:0<6}",
        &value[..10],
        &value[11..DATE_TIME_LEN]
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aware_utc_round_trips() {
        let naive = "2024-01-02 03:04:05.678000";
        assert_eq!(
            naive_utc_to_aware_utc(naive),
            "2024-01-02T03:04:05.678000+00:00"
        );
        assert_eq!(
            journal_datetime_to_naive_utc(&naive_utc_to_aware_utc(naive)),
            naive
        );
    }

    #[test]
    fn now_round_trips() {
        let aware = now_aware_utc();
        assert!(aware.ends_with("+00:00"), "{aware}");
        assert_eq!(aware.as_bytes()[10], b'T', "{aware}");
        let naive = journal_datetime_to_naive_utc(&aware);
        assert_eq!(naive_utc_to_aware_utc(&naive), aware);
    }

    #[test]
    fn accepts_the_shapes_optuna_writes() {
        for input in [
            "2024-01-02T03:04:05.678000+00:00",
            "2024-01-02T03:04:05.678000Z",
            "2024-01-02t03:04:05.678000z",
            "2024-01-02 03:04:05.678000+00:00",
            // Logs predating aware UTC.
            "2024-01-02 03:04:05.678000",
        ] {
            assert_eq!(
                journal_datetime_to_naive_utc(input),
                "2024-01-02 03:04:05.678000",
                "{input}"
            );
        }
    }

    #[test]
    fn normalizes_fractional_precision() {
        for (input, expected) in [
            ("2024-01-02T03:04:05Z", "2024-01-02 03:04:05.000000"),
            ("2024-01-02T03:04:05.5Z", "2024-01-02 03:04:05.500000"),
            ("2024-01-02T03:04:05.06Z", "2024-01-02 03:04:05.060000"),
            ("2024-01-02T03:04:05.007Z", "2024-01-02 03:04:05.007000"),
            ("2024-01-02T03:04:05.000008Z", "2024-01-02 03:04:05.000008"),
            // Finer than microseconds is truncated, as Python's datetime does.
            (
                "2024-01-02T03:04:05.123456789Z",
                "2024-01-02 03:04:05.123456",
            ),
        ] {
            assert_eq!(journal_datetime_to_naive_utc(input), expected, "{input}");
        }
    }

    #[test]
    fn an_offset_is_dropped_rather_than_applied() {
        // Documented limitation: nothing Rustuna or Optuna writes has a non-UTC offset, and
        // honouring one would mean doing date arithmetic here.
        assert_eq!(
            journal_datetime_to_naive_utc("2024-01-02T03:04:05.678000+09:00"),
            "2024-01-02 03:04:05.678000"
        );
        assert_eq!(
            journal_datetime_to_naive_utc("2024-01-02T03:04:05.678000-05:30"),
            "2024-01-02 03:04:05.678000"
        );
    }

    #[test]
    fn values_that_are_not_datetimes_are_left_alone() {
        for input in [
            "",
            "not-a-datetime",
            "2024-01-02",
            "2024/01/02T03:04:05Z",
            "2024-01-02X03:04:05Z",
            "2024-01-02T03-04:05Z",
        ] {
            assert_eq!(journal_datetime_to_naive_utc(input), input, "{input}");
        }
    }
}
