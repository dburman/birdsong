//! Stable names for ALSA capture devices, read from `/proc/asound` (no ALSA library).
//!
//! Card numbers are assigned at boot and can change: after a kernel upgrade a USB microphone moved
//! from card 1 to card 3. `plughw:CARD=<id>,DEV=<n>` names the same device on every boot.

use std::path::Path;

/// One capture device the kernel reports.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CaptureDevice {
    /// Card number (can change between boots).
    pub card: u32,
    pub device: u32,
    /// Card id, such as `Device` (stable).
    pub card_id: String,
    /// Card description, such as `USB Audio Device`.
    pub card_name: String,
}

impl CaptureDevice {
    /// `plughw:CARD=<id>,DEV=<n>`: stable, and converts rate and channel count as needed.
    pub fn stable_name(&self) -> String {
        format!("plughw:CARD={},DEV={}", self.card_id, self.device)
    }
}

/// Capture devices on this machine; empty where `/proc/asound` does not exist.
pub fn capture_devices() -> Vec<CaptureDevice> {
    capture_devices_in(Path::new("/proc/asound"))
}

fn capture_devices_in(dir: &Path) -> Vec<CaptureDevice> {
    match (
        std::fs::read_to_string(dir.join("cards")),
        std::fs::read_to_string(dir.join("pcm")),
    ) {
        (Ok(cards), Ok(pcm)) => parse(&cards, &pcm),
        _ => Vec::new(),
    }
}

/// Parse `/proc/asound/cards` and `/proc/asound/pcm`.
pub fn parse(cards: &str, pcm: &str) -> Vec<CaptureDevice> {
    // " 3 [Device         ]: USB-Audio - USB Audio Device"
    let card_info = |card: u32| {
        cards.lines().find_map(|line| {
            let (number, rest) = line.trim_start().split_once(' ')?;
            if number.parse::<u32>().ok()? != card {
                return None;
            }
            let id = rest
                .trim_start()
                .strip_prefix('[')?
                .split_once(']')?
                .0
                .trim();
            let name = rest.split_once(" - ").map_or(id, |(_, name)| name.trim());
            Some((id.to_string(), name.to_string()))
        })
    };
    // "03-00: USB Audio : USB Audio : playback 1 : capture 1"
    pcm.lines()
        .filter(|line| line.contains("capture"))
        .filter_map(|line| {
            let (card, rest) = line.split_once('-')?;
            let device = rest.split_once(':')?.0;
            let card = card.trim().parse().ok()?;
            let device = device.trim().parse().ok()?;
            let (card_id, card_name) = card_info(card)?;
            Some(CaptureDevice {
                card,
                device,
                card_id,
                card_name,
            })
        })
        .collect()
}

/// For a device named by card number (`hw:3,0`, `plughw:3`), the card and device numbers.
pub fn numbered_card(device: &str) -> Option<(u32, u32)> {
    let rest = device
        .strip_prefix("plughw:")
        .or_else(|| device.strip_prefix("hw:"))?;
    let (card, dev) = rest.split_once(',').unwrap_or((rest, "0"));
    Some((card.trim().parse().ok()?, dev.trim().parse().ok()?))
}

/// A warning for a device named by card number, suggesting the stable name when the card is
/// present. `None` for names that are already stable.
pub fn unstable_name_warning(device: &str, devices: &[CaptureDevice]) -> Option<String> {
    let (card, dev) = numbered_card(device)?;
    let found = devices.iter().find(|d| d.card == card && d.device == dev);
    Some(match found {
        Some(d) => format!(
            "device {device:?} names the card by number, which can change between boots; \
             use \"{}\" ({})",
            d.stable_name(),
            d.card_name
        ),
        None => format!(
            "device {device:?} names the card by number, which can change between boots, and no \
             capture device is card {card} device {dev} now; run `birdsong devices` for stable names"
        ),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const CARDS: &str = " 0 [Headphones     ]: bcm2835_headpho - bcm2835 Headphones
                      bcm2835 Headphones
 3 [Device         ]: USB-Audio - USB Audio Device
                      C-Media Electronics Inc. USB Audio Device at usb-0000:01:00.0-1.1, full speed
";
    const PCM: &str = "00-00: bcm2835 Headphones : bcm2835 Headphones : playback 8
03-00: USB Audio : USB Audio : playback 1 : capture 1
";

    #[test]
    fn capture_devices_from_proc() {
        let devices = parse(CARDS, PCM);
        assert_eq!(
            devices,
            [CaptureDevice {
                card: 3,
                device: 0,
                card_id: "Device".into(),
                card_name: "USB Audio Device".into(),
            }]
        );
        assert_eq!(devices[0].stable_name(), "plughw:CARD=Device,DEV=0");
    }

    #[test]
    fn numbered_names_get_a_stable_suggestion() {
        let devices = parse(CARDS, PCM);
        assert_eq!(numbered_card("hw:3,0"), Some((3, 0)));
        assert_eq!(numbered_card("plughw:3"), Some((3, 0)));
        assert_eq!(numbered_card("plughw:CARD=Device,DEV=0"), None);
        assert_eq!(numbered_card("default"), None);
        let warning = unstable_name_warning("hw:3,0", &devices).unwrap();
        assert!(warning.contains("plughw:CARD=Device,DEV=0"), "{warning}");
        let missing = unstable_name_warning("hw:1,0", &devices).unwrap();
        assert!(missing.contains("birdsong devices"), "{missing}");
        assert!(unstable_name_warning("plughw:CARD=Device,DEV=0", &devices).is_none());
    }

    #[test]
    fn missing_proc_gives_no_devices() {
        assert!(capture_devices_in(Path::new("/nonexistent")).is_empty());
    }
}
