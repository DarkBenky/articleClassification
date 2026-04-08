import json
import csv

# ── Build FIPS lookup tables ──────────────────────────────────────────────────
flipCodeToName = {}
isoToFlipCode  = {}
nameToFlipCode = {}
validFlipCodes = set()

with open("fipsCodes.csv", "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        fips = row["FIPS 10-4"].strip()
        iso  = row["ISO 3166"].strip()
        name = row["Name"].strip()

        if fips and fips != "-":
            validFlipCodes.add(fips)
            flipCodeToName[fips] = name
            nameToFlipCode[name.lower()] = fips

        if iso and iso != "-":
            isoToFlipCode[iso] = fips if (fips and fips != "-") else None

# ── Alias table: country name variants and nationality demonyms → FIPS ────────
ALIAS_TO_FIPS = {
    # country aliases / alternate spellings
    "russia": "RS", "britain": "UK", "great britain": "UK",
    "united kingdom": "UK", "united states": "US",
    "south korea": "KS", "north korea": "KN",
    "china": "CH", "iran": "IR", "iraq": "IZ", "syria": "SY",
    "india": "IN", "australia": "AS", "italy": "IT", "france": "FR",
    "germany": "GM", "spain": "SP", "japan": "JA", "canada": "CA",
    "brazil": "BR", "mexico": "MX", "west bank": "WE",
    "gaza strip": "GZ", "palestine": "WE",
    "scotland": "UK", "wales": "UK", "england": "UK",
    "northern ireland": "UK",
    "holy see": "VT", "jersey": "JE", "holland": "NL",
    "vietnam": "VM", "afghanistan": "AF", "nigeria": "NI",
    "israel": "IS", "jordan": "JO", "kuwait": "KU",
    "thailand": "TH", "turkey": "TU", "libya": "LY",
    "venezuela": "VE", "chile": "CI", "colombia": "CO",
    "mali": "ML", "ghana": "GH", "indonesia": "ID",
    "malaysia": "MY", "bangladesh": "BG", "pakistan": "PK",
    "saudi arabia": "SA", "bahrain": "BA", "singapore": "SN",
    "new zealand": "NZ", "ukraine": "UP", "ireland": "EI",
    "austria": "AU", "switzerland": "SZ", "malta": "MT",
    "cyprus": "CY", "chad": "CD", "cameroon": "CM",
    "yemen": "YM", "ethiopia": "ET", "kenya": "KE",
    "uganda": "UG", "tanzania": "TZ", "mozambique": "MZ",
    "zambia": "ZA", "zimbabwe": "ZI", "namibia": "WA",
    "botswana": "BC", "senegal": "SG", "ivory coast": "IV",
    "cote d'ivoire": "IV", "sierra leone": "SL", "liberia": "LI",
    "somalia": "SO", "eritrea": "ER", "sudan": "SU",
    "south sudan": "OD", "rwanda": "RW", "burundi": "BY",
    "angola": "AO", "madagascar": "MA", "malawi": "MI",
    "burkina faso": "UV", "guinea": "GV", "guinea-bissau": "PU",
    "djibouti": "DJ", "gambia": "GA", "cape verde": "CV",
    "comoros": "CN", "equatorial guinea": "EK", "gabon": "GB",
    "central african republic": "CT", "kosovo": "KV",
    "democratic republic of the congo": "CG",
    "republic of the congo": "CF", "sao tome and principe": "TP",
    "antigua and barbuda": "AC", "bahamas": "BF",
    "barbados": "BB", "belize": "BH", "bolivia": "BL",
    "cambodia": "CB", "costa rica": "CS", "croatia": "HR",
    "cuba": "CU", "dominican republic": "DR", "ecuador": "EC",
    "el salvador": "ES", "fiji": "FJ", "guyana": "GY",
    "haiti": "HA", "honduras": "HO", "jamaica": "JM",
    "laos": "LA", "nepal": "NP", "nicaragua": "NU",
    "panama": "PM", "papua new guinea": "PP", "paraguay": "PA",
    "peru": "PE", "philippines": "RP", "sri lanka": "CE",
    "suriname": "NS", "timor-leste": "TT", "tonga": "TN",
    "trinidad and tobago": "TD", "tuvalu": "TV", "vanuatu": "NH",
    "taiwan": "TW", "hong kong": "HK", "macau": "MC",
    "antarctica": "AY", "gold coast": "AS",  # Gold Coast city, Queensland
    # demonyms (singular and plural)
    "american": "US", "americans": "US",
    "russian": "RS", "russians": "RS",
    "chinese": "CH",
    "italian": "IT",
    "australian": "AS", "australians": "AS", "aussie": "AS",
    "british": "UK",
    "spanish": "SP",
    "irish": "EI",
    "canadian": "CA",
    "brazilian": "BR",
    "indonesian": "ID",
    "iranian": "IR",
    "thai": "TH",
    "nigerian": "NI",
    "malaysian": "MY",
    "syrian": "SY",
    "north korean": "KN",
    "egyptian": "EG",
    "french": "FR",
    "korean": "KS", "south korean": "KS",
    "ethiopian": "ET",
    "jordanian": "JO", "jordanians": "JO",
    "kuwaiti": "KU",
    "bahraini": "BA",
    "chilean": "CI", "chileans": "CI",
    "swedish": "SW", "swedes": "SW",
    "guatemalan": "GT",
    "salvadoran": "ES",
    "german": "GM",
    "japanese": "JA",
    "pakistani": "PK",
    "indian": "IN",
    "central african": "CT",
    "venezuelan": "VE",
    "greek": "GR",
    "portuguese": "PO",
    "polish": "PL",
    "romanian": "RO",
    "ukrainian": "UP",
    "czech": "EZ",
    "hungarian": "HU",
    "norwegian": "NO",
    "danish": "DA",
    "finnish": "FI",
    "swiss": "SZ",
    "austrian": "AU",
    "belgian": "BE",
    "dutch": "NL",
    "turkish": "TU",
    "saudi": "SA",
    "emirati": "AE",
    "qatari": "QA",
    "yemeni": "YM",
    "libyan": "LY",
    "moroccan": "MO",
    "algerian": "AG",
    "tunisian": "TS",
    "sudanese": "SU",
    "somali": "SO",
    "kenyan": "KE",
    "ugandan": "UG",
    "tanzanian": "TZ",
    "ghanaian": "GH",
    "zimbabwean": "ZI",
    "zambian": "ZA",
    "angolan": "AO",
    "congolese": "CG",
    "cambodian": "CB",
    "vietnamese": "VM",
    "filipino": "RP",
    "sri lankan": "CE",
    "bangladeshi": "BG",
    "nepalese": "NP",
    "singaporean": "SN",
    "new zealander": "NZ",
    "iraqi": "IZ",
}

UNKNOWN_LABEL = "Unknown"


def resolve_to_fips(key: str) -> str:
    """Map an arbitrary location string to a FIPS code, or UNKNOWN_LABEL."""
    # Already a valid FIPS code
    if key in validFlipCodes:
        return key

    # Explicit unknown / null entries
    if key in ("Unknown", "null"):
        return UNKNOWN_LABEL

    # Check alias/demonym table (case-insensitive)
    lower = key.lower()
    if lower in ALIAS_TO_FIPS:
        return ALIAS_TO_FIPS[lower]

    # Try direct name lookup in FIPS CSV
    fips = nameToFlipCode.get(lower)
    if fips:
        return fips

    # Multi-part location like "City, State/Region, Country"
    # Walk from most-specific (country last) backward
    parts = [p.strip() for p in key.split(",")]
    for part in reversed(parts):
        part_lower = part.lower()
        if part_lower in ALIAS_TO_FIPS:
            return ALIAS_TO_FIPS[part_lower]
        fips = nameToFlipCode.get(part_lower)
        if fips:
            return fips
        # Maybe the part itself is an ISO-2 code
        if part in isoToFlipCode and isoToFlipCode[part]:
            return isoToFlipCode[part]

    return UNKNOWN_LABEL


if __name__ == "__main__":
    unique_locations = json.load(open("unique_locations.json", "r"))

    # Build location_key → FIPS mapping
    location_to_fips = {}
    for key in unique_locations:
        location_to_fips[key] = resolve_to_fips(key)

    # Aggregate original counts per FIPS code
    fips_counts: dict[str, int] = {}
    for key, count in unique_locations.items():
        fips = location_to_fips[key]
        fips_counts[fips] = fips_counts.get(fips, 0) + count

    # Sort by count descending
    fips_counts = dict(sorted(fips_counts.items(), key=lambda x: x[1], reverse=True))

    # Save outputs
    with open("location_to_fips.json", "w") as f:
        json.dump(location_to_fips, f, indent=2)

    with open("unique_fips_locations.json", "w") as f:
        json.dump(fips_counts, f, indent=2)

    print(f"Categories reduced: {len(unique_locations)} → {len(fips_counts)}")
    print(f"\nTop 20 FIPS categories by count:")
    for code, cnt in list(fips_counts.items())[:20]:
        name = flipCodeToName.get(code, code)
        print(f"  {code:4s}  {name:35s}  {cnt:>12,}")

    unresolved = sum(1 for v in location_to_fips.values() if v == UNKNOWN_LABEL)
    unresolved_count = fips_counts.get(UNKNOWN_LABEL, 0)
    total = sum(unique_locations.values())
    print(f"\nUnresolved keys : {unresolved} distinct → {unresolved_count:,} samples ({unresolved_count/total*100:.1f}% of total)")
    print(f"\nSaved: location_to_fips.json  |  unique_fips_locations.json")
