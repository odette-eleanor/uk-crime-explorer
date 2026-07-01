import requests
import pandas as pd
from pathlib import Path

# --- Config ---
# We'll use Leeds city centre as our starting point
# Lat/Lng for Leeds city centre
LAT = 53.7996
LNG = -1.5491

# Months to fetch (YYYY-MM format) — last 12 months
MONTHS = [
    "2024-01", "2024-02", "2024-03", "2024-04",
    "2024-05", "2024-06", "2024-07", "2024-08",
    "2024-09", "2024-10", "2024-11", "2024-12",
]

BASE_URL = "https://data.police.uk/api/crimes-street/all-crime"
DATA_DIR = Path("data")


def fetch_crimes_for_month(lat: float, lng: float, month: str) -> list[dict]:
    """Fetch all street crimes for a given location and month."""
    params = {
        "lat": lat,
        "lng": lng,
        "date": month,
    }
    response = requests.get(BASE_URL, params=params, timeout=30)

    if response.status_code == 200:
        print(f"  ✓ {month} — {len(response.json())} crimes fetched")
        return response.json()
    else:
        print(f"  ✗ {month} — failed ({response.status_code})")
        return []


def ingest_all(lat: float, lng: float, months: list[str]) -> pd.DataFrame:
    """Fetch crimes for all months and combine into a single DataFrame."""
    all_records = []

    for month in months:
        records = fetch_crimes_for_month(lat, lng, month)
        all_records.extend(records)

    if not all_records:
        print("No data fetched. Check your internet or try different months.")
        return pd.DataFrame()

    df = pd.DataFrame(all_records)
    return df


def flatten_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """The API returns nested dicts — flatten them into clean columns."""
    # location is a nested dict: {latitude, longitude, street: {id, name}}
    df["latitude"] = df["location"].apply(lambda x: x["latitude"] if x else None)
    df["longitude"] = df["location"].apply(lambda x: x["longitude"] if x else None)
    df["street_name"] = df["location"].apply(
        lambda x: x["street"]["name"] if x and "street" in x else None
    )

    # outcome_status is nested: {category, date} or None
    df["outcome_category"] = df["outcome_status"].apply(
        lambda x: x["category"] if x else "Unknown"
    )

    # Drop the original nested columns
    df = df.drop(columns=["location", "outcome_status", "context"])

    # Convert lat/lng to float
    df["latitude"] = df["latitude"].astype(float)
    df["longitude"] = df["longitude"].astype(float)

    return df


def main():
    print(f"Fetching crime data for lat={LAT}, lng={LNG}")
    print(f"Months: {MONTHS[0]} to {MONTHS[-1]}\n")

    # Fetch
    df_raw = ingest_all(LAT, LNG, MONTHS)

    if df_raw.empty:
        return

    print(f"\nTotal records fetched: {len(df_raw)}")

    # Flatten nested columns
    df_clean = flatten_dataframe(df_raw)

    # Save to CSV
    output_path = DATA_DIR / "crimes_raw.csv"
    df_clean.to_csv(output_path, index=False)
    print(f"\nSaved to {output_path}")
    print(f"Columns: {list(df_clean.columns)}")
    print(f"Shape: {df_clean.shape}")


if __name__ == "__main__":
    main()