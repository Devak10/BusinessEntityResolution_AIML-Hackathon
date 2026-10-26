"""
Data Preprocessing Module for Business Entity Resolution (Step 4)
=================================================================

Provides deterministic, reproducible text normalization functions for:
- Business Names (lowercasing, punctuation removal, synthetic marker removal, legal suffix normalization)
- Addresses (lowercasing, punctuation removal, digit preservation, spatial token normalization)
- Countries (lowercasing, trimming, dynamic support for any country)

Crucial Principles:
- Original columns (entity_id, business_name, business_address, country) are PRESERVED.
- New normalized columns (business_name_normalized, business_address_normalized, country_normalized) are ADDED.
- No ground truth labels or external lookups are used (Zero Data Leakage).
"""

import re
import unicodedata
import pandas as pd


def normalize_text(text: str) -> str:
    """
    General text normalization helper:
    - Safe string conversion
    - Unicode decomposition (NFKD)
    - Lowercasing
    - Stripping and collapsing repeated whitespace
    """
    if text is None:
        return ""
    s = str(text)
    # Decompose unicode characters to standard ASCII where possible
    s = unicodedata.normalize("NFKD", s)
    s = s.lower().strip()
    # Collapse multiple whitespaces
    s = re.sub(r"\s+", " ", s)
    return s


def remove_synthetic_markers(text: str) -> str:
    """
    Removes artificial dataset generation markers ('distractor', 'scraped')
    case-insensitively, ensuring they do not act as spurious ML shortcuts.
    """
    # Remove standalone words 'distractor' and 'scraped'
    s = re.sub(r"\bdistractor\b", "", text, flags=re.IGNORECASE)
    s = re.sub(r"\bscraped\b", "", s, flags=re.IGNORECASE)
    return s


def normalize_business_name(name: str) -> str:
    """
    Normalizes business names for entity resolution:
    1. Lowercase and unicode normalization
    2. Removal of synthetic generator markers (distractor, scraped)
    3. Removal of punctuation while preserving alphanumeric characters
    4. Canonicalization of common legal suffixes (Pvt Ltd, Ltd, Inc, LLC, Corp, Co)
    5. Normalization of common business keywords (tech, intl, mfg, svcs, solns)
    6. Stripping of redundant leading 'the '
    7. Whitespace normalization
    """
    if name is None:
        return ""
    s = normalize_text(name)

    # 1. Remove synthetic generator markers
    s = remove_synthetic_markers(s)

    # 2. Standardize ampersand
    s = re.sub(r"&", " and ", s)

    # 3. Clean punctuation into spaces to separate tokens cleanly
    s = re.sub(r"[^\w\s]", " ", s)

    # 4. Canonicalize multi-word and compound legal suffixes first
    # Private Limited variations
    s = re.sub(r"\b(private\s+limited|private\s+ltd|pvt\s+limited|p\s+limited|pvt\s+ltd|p\s+ltd|p\s+l\s*t\s*d)\b", "pvt ltd", s)
    # Limited Liability Company variations
    s = re.sub(r"\b(limited\s+liability\s+company|limited\s+liability\s+co|l\s*l\s*c)\b", "llc", s)
    # Limited Liability Partnership
    s = re.sub(r"\b(limited\s+liability\s+partnership|l\s*l\s*p)\b", "llp", s)
    # Limited Partnership
    s = re.sub(r"\b(limited\s+partnership|l\s*p)\b", "lp", s)

    # 5. Canonicalize single-word legal suffixes
    s = re.sub(r"\b(corporation|corp)\b", "corp", s)
    s = re.sub(r"\b(incorporated|inc)\b", "inc", s)
    s = re.sub(r"\b(limited|ltd)\b", "ltd", s)
    s = re.sub(r"\b(company|co)\b", "co", s)

    # 6. Normalize common domain word abbreviations
    s = re.sub(r"\b(technologies|technology)\b", "tech", s)
    s = re.sub(r"\binternational\b", "intl", s)
    s = re.sub(r"\bmanufacturing\b", "mfg", s)
    s = re.sub(r"\bservices\b", "svcs", s)
    s = re.sub(r"\bsolutions\b", "solns", s)
    s = re.sub(r"\bmanagement\b", "mgmt", s)
    s = re.sub(r"\bdepartment\b", "dept", s)

    # 7. Remove leading 'the '
    s = re.sub(r"^the\s+", "", s)

    # 8. Collapse whitespace
    s = re.sub(r"\s+", " ", s).strip()
    return s


def normalize_address(address: str) -> str:
    """
    Normalizes commercial and postal addresses:
    1. Lowercase and unicode normalization
    2. Digits are strictly PRESERVED (street numbers, suites, floors, PIN/ZIP codes)
    3. Removal of punctuation while preserving alphanumeric tokens
    4. Canonicalization of spatial tokens and abbreviations:
       - road -> rd, street -> st, avenue -> ave, boulevard -> blvd
       - suite -> ste, building -> bldg, apartment -> apt, floor -> fl
       - highway -> hwy, parkway -> pkwy, expressway -> expwy
       - industrial area -> indl area, sector -> sec, phase -> ph, tower -> twr
    5. Whitespace normalization
    """
    if address is None:
        return ""
    s = normalize_text(address)

    # Replace punctuation characters with space (preserves digits and letters)
    s = re.sub(r"[^\w\s]", " ", s)

    # Canonicalize spatial abbreviations
    s = re.sub(r"\b(road|rd)\b", "rd", s)
    s = re.sub(r"\b(street|st)\b", "st", s)
    s = re.sub(r"\b(avenue|ave)\b", "ave", s)
    s = re.sub(r"\b(boulevard|blvd)\b", "blvd", s)
    s = re.sub(r"\b(suite|ste)\b", "ste", s)
    s = re.sub(r"\b(building|bldg)\b", "bldg", s)
    s = re.sub(r"\b(apartment|apt)\b", "apt", s)
    s = re.sub(r"\b(floor|flr|fl)\b", "fl", s)
    s = re.sub(r"\b(highway|hwy)\b", "hwy", s)
    s = re.sub(r"\b(parkway|pkwy)\b", "pkwy", s)
    s = re.sub(r"\b(expressway|expwy)\b", "expwy", s)
    s = re.sub(r"\b(industrial\s+area|indl\s+area)\b", "indl area", s)
    s = re.sub(r"\bsector\b", "sec", s)
    s = re.sub(r"\bphase\b", "ph", s)
    s = re.sub(r"\btower\b", "twr", s)
    s = re.sub(r"\bblock\b", "blk", s)

    # Collapse whitespace
    s = re.sub(r"\s+", " ", s).strip()
    return s


def normalize_country(country: str) -> str:
    """
    Normalizes country strings:
    - Lowercase, trim, and collapse spaces
    - Dynamic support for any country without hardcoding
    - Standardizes common country acronyms
    """
    if country is None:
        return ""
    s = normalize_text(country)

    # Standardize common aliases
    if s in ["usa", "united states", "united states of america", "u s a", "u s"]:
        return "usa"
    if s in ["india", "republic of india"]:
        return "india"

    return s


def preprocess_source_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies the full preprocessing pipeline to a source DataFrame.

    Preserves original columns:
    - entity_id
    - business_name
    - business_address
    - country

    Adds new normalized columns:
    - business_name_normalized
    - business_address_normalized
    - country_normalized
    """
    out = df.copy()

    # Create new normalized columns
    out["business_name_normalized"] = out["business_name"].astype(str).apply(normalize_business_name)
    out["business_address_normalized"] = out["business_address"].astype(str).apply(normalize_address)
    out["country_normalized"] = out["country"].astype(str).apply(normalize_country)

    return out
