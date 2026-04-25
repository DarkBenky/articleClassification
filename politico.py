from datasets import load_dataset
from addArticles import addToDataset
from groupLocations import resolve_to_fips

# ds = load_dataset("SinclairSchneider/politico_eu")

# c = 0
# len_ds = len(ds["train"])
# for article in ds["train"]:
#     text = article['title'] + " " + article['content']
#     location = article['countries'][0] if article['countries'] else "Unknown"
#     category = article['category']
#     fips_location = resolve_to_fips(location)
#     addToDataset({
#         "text": text,
#         "location": fips_location,
#         "category": category,
#     })
#     if c % 100 == 0:
#         print(f"Processed {c}/{len_ds} articles...")
#     c += 1

import pandas as pd
df = pd.read_csv("RussianInvasion.csv")

for _, row in df.iterrows():
    test = row['Text Text']
    location = row['location']
    if "ukraine" in location.lower().strip():
        location = "Ukraine"
    elif "russia" in location.lower().strip():
        location = "Russia"
    else:
        continue
    category = "Russian invasion"
    fips_location = resolve_to_fips(location)
    addToDataset({
        "text": test,
        "location": fips_location,
        "category": category,
    })
    
