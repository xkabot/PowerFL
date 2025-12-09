import pandas as pd
import glob
import os

# Input + Output paths
DATA_FOLDER = "src/reddit-dataset"
OUTPUT_PATH = "data/test.csv"

# get columns from data set
COLS = [
    "index", "text", "id", "subreddit", "meta", "time", "author", "ups", "downs",
    "authorlinkkarma", "authorkarma", "authorisgold"
]

# ouput cols for cleaned data (remove text)
OUTPUT_COLS = [
    "id", "subreddit", "meta", "time", "author", "ups", "downs",
    "authorlinkkarma", "authorkarma", "authorisgold"
]

# Find all CSV files
csv_files = glob.glob(os.path.join(DATA_FOLDER, "*.csv"))

def load_csv_clean(path):
    """
    Load CSV, skip the numeric header row, and assign correct column names.
    """
    try:
        df = pd.read_csv(path, skiprows=1, header=None, names=COLS, low_memory=False)
        
        # Filter out rows where subreddit is numeric (these are leftover header rows)
        # Real subreddit names are strings like "community", "funny", etc.
        df = df[~df["subreddit"].astype(str).str.match(r'^\d+\.?\d*$')]
        
        return df
    except Exception as e:
        print(f"Failed reading {path}: {e}")
        print("Check if it is downloaded through curl command in readme")
        return pd.DataFrame(columns=COLS)

# Load each CSV
dfs = [load_csv_clean(f) for f in csv_files]

# Combine
combined = pd.concat(dfs, ignore_index=True)

# Remove empty text
combined = combined[combined["text"].notna()]

# Sample up to 1000 per subreddit (to get ~50k total if 50 subs)
combined = (
    combined
    .groupby("subreddit", group_keys=False)
    .apply(lambda g: g.sample(min(1000, len(g)), random_state=42))
    .reset_index(drop=True)
)

# Select only output columns (drop index and text)
combined = combined[OUTPUT_COLS]

# Save result
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
combined.to_csv(OUTPUT_PATH, index=False)

print("Final shape:", combined.shape)
print("Saved to:", OUTPUT_PATH)