//! Where to read about a detected species: Cornell Lab's All About Birds for birds, and
//! iNaturalist for other animals.
//!
//! All About Birds pages are addressed by common name (`Coopers_Hawk`); a name it does not have
//! lands on its search results. The BirdNET Geomodel's species codes tell birds (eBird codes) from
//! other animals, whose codes are iNaturalist taxon ids, so those link to the exact taxon page.

/// `https://www.allaboutbirds.org/guide/Coopers_Hawk/overview` for "Cooper's Hawk".
pub fn all_about_birds(common_name: &str) -> String {
    let slug: String = common_name
        .chars()
        .filter(|c| !matches!(c, '\'' | '\u{2019}'))
        .map(|c| if c == ' ' { '_' } else { c })
        .collect();
    format!(
        "https://www.allaboutbirds.org/guide/{}/overview",
        percent_encode(&slug)
    )
}

/// The iNaturalist page of a taxon id (the Geomodel's code for animals other than birds).
pub fn inaturalist_taxon(id: &str) -> String {
    format!("https://www.inaturalist.org/taxa/{id}")
}

/// iNaturalist's taxon search, for species without a known id.
pub fn inaturalist_search(scientific_name: &str) -> String {
    format!(
        "https://www.inaturalist.org/taxa/search?q={}",
        percent_encode(scientific_name)
    )
}

/// Percent-encode everything but unreserved URL characters.
fn percent_encode(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for b in text.bytes() {
        if b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_' | b'.' | b'~') {
            out.push(b as char);
        } else {
            out.push_str(&format!("%{b:02X}"));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_about_birds_slugs() {
        assert_eq!(
            all_about_birds("Black-capped Chickadee"),
            "https://www.allaboutbirds.org/guide/Black-capped_Chickadee/overview"
        );
        assert_eq!(
            all_about_birds("Cooper's Hawk"),
            "https://www.allaboutbirds.org/guide/Coopers_Hawk/overview"
        );
        assert_eq!(
            all_about_birds("Chuck-will\u{2019}s-widow"),
            "https://www.allaboutbirds.org/guide/Chuck-wills-widow/overview"
        );
        assert!(all_about_birds("Rüppell's Warbler").contains("R%C3%BCppells_Warbler"));
    }

    #[test]
    fn inaturalist_links() {
        assert_eq!(
            inaturalist_taxon("42069"),
            "https://www.inaturalist.org/taxa/42069"
        );
        assert_eq!(
            inaturalist_search("Vulpes vulpes"),
            "https://www.inaturalist.org/taxa/search?q=Vulpes%20vulpes"
        );
    }
}
