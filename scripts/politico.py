"""Import the Russian-invasion article set into the corpus."""

import pandas as pd

from addArticles import addToDataset
from groupLocations import resolve_to_fips
from project_config import RUSSIAN_INVASION_PATH


def importRussianInvasion():
    """Append Russian/Ukrainian rows from RussianInvasion.csv to the corpus."""
    df = pd.read_csv(RUSSIAN_INVASION_PATH)

    for _, row in df.iterrows():
        location = row["location"]
        if not isinstance(location, str):
            continue
        if "ukraine" in location.lower().strip():
            location = "Ukraine"
        elif "russia" in location.lower().strip():
            location = "Russia"
        else:
            continue
        addToDataset({
            "text": row["Text Text"],
            "location": resolve_to_fips(location),
            "category": "Russian invasion",
        })


if __name__ == "__main__":
    importRussianInvasion()
    
