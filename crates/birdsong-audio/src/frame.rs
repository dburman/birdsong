use birdsong_core::SAMPLE_RATE_HZ;
use chrono::{DateTime, TimeDelta, Utc};

/// A block of mono 48 kHz samples from one source.
#[derive(Clone, Debug, PartialEq)]
pub struct AudioFrame {
    pub samples: Vec<f32>,
    /// UTC time of the first sample.
    pub captured_at: DateTime<Utc>,
}

/// Decibels to a linear amplitude factor.
pub fn db_to_gain(db: f32) -> f32 {
    10f32.powf(db / 20.0)
}

/// Multiply in place; a no-op for unity gain.
pub fn apply_gain(samples: &mut [f32], gain: f32) {
    if gain != 1.0 {
        for s in samples {
            *s *= gain;
        }
    }
}

const NANOS_PER_SECOND: i128 = 1_000_000_000;

fn div_round(num: i128, den: i128) -> i128 {
    if num >= 0 {
        (num + den / 2) / den
    } else {
        (num - den / 2) / den
    }
}

/// Duration of `n` samples at 48 kHz, rounded to the nearest nanosecond.
pub fn samples_to_delta(n: u64) -> TimeDelta {
    let nanos = div_round(n as i128 * NANOS_PER_SECOND, SAMPLE_RATE_HZ as i128);
    TimeDelta::nanoseconds(nanos.min(i64::MAX as i128) as i64)
}

/// Number of 48 kHz samples in a duration, rounded to the nearest sample (may be negative).
pub fn delta_to_samples(d: TimeDelta) -> i64 {
    let nanos = d.num_nanoseconds().unwrap_or(if d < TimeDelta::zero() {
        i64::MIN
    } else {
        i64::MAX
    });
    let samples = div_round(nanos as i128 * SAMPLE_RATE_HZ as i128, NANOS_PER_SECOND);
    samples.clamp(i64::MIN as i128, i64::MAX as i128) as i64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gain_conversion() {
        assert_eq!(db_to_gain(0.0), 1.0);
        assert!((db_to_gain(20.0) - 10.0).abs() < 1e-4);
        assert!((db_to_gain(-6.0206) - 0.5).abs() < 1e-4);
        let mut s = [0.5, -0.25];
        apply_gain(&mut s, 2.0);
        assert_eq!(s, [1.0, -0.5]);
    }

    #[test]
    fn sample_time_round_trip() {
        assert_eq!(samples_to_delta(48_000), TimeDelta::seconds(1));
        assert_eq!(samples_to_delta(144_000), TimeDelta::seconds(3));
        assert_eq!(samples_to_delta(1), TimeDelta::nanoseconds(20_833));
        for n in [0u64, 1, 7, 4_800, 144_000, 4_320_000, 123_456_789_012] {
            assert_eq!(delta_to_samples(samples_to_delta(n)), n as i64, "n={n}");
        }
        assert_eq!(delta_to_samples(TimeDelta::milliseconds(-500)), -24_000);
    }
}
