import os
import math
import time
import requests
import pandas as pd

# -----------------------------
# CONFIG
# -----------------------------
ZOOM = 18
HEADERS = {
    "User-Agent": "Satellite-Imagery-Based-Property-Valuation/1.0 (academic use)"
}

ESRI_URL = (
    "https://services.arcgisonline.com/ArcGIS/rest/services/"
    "World_Imagery/MapServer/tile"
)

# -----------------------------
# LAT/LON → TILE CONVERSION
# -----------------------------
def latlon_to_tile(lat, lon, zoom):
    lat_rad = math.radians(lat)
    n = 2.0 ** zoom

    xtile = int((lon + 180.0) / 360.0 * n)
    ytile = int(
        (1.0 - math.log(math.tan(lat_rad) + (1 / math.cos(lat_rad))) / math.pi)
        / 2.0 * n
    )

    return xtile, ytile

# -----------------------------
# IMAGE DOWNLOAD
# -----------------------------
def fetch_satellite_image(lat, lon, save_path, zoom=ZOOM):
    if os.path.exists(save_path):
        return True  # already downloaded

    x, y = latlon_to_tile(lat, lon, zoom)
    url = f"{ESRI_URL}/{zoom}/{y}/{x}"

    try:
        response = requests.get(url, headers=HEADERS, timeout=15)

        if response.status_code == 200:
            with open(save_path, "wb") as f:
                f.write(response.content)
            return True
        else:
            print("HTTP error:", response.status_code)
            return False

    except Exception as e:
        print("Request failed:", e)
        return False

# -----------------------------
# BATCH DOWNLOADER
# -----------------------------
def download_images(csv_path, image_dir):
    os.makedirs(image_dir, exist_ok=True)
    df = pd.read_csv(csv_path)

    print(f"Total rows: {len(df)}")

    for idx, row in df.iterrows():
        save_path = os.path.join(image_dir, f"{idx}.png")

        success = fetch_satellite_image(
            row["lat"],
            row["long"],
            save_path
        )

        if not success:
            print(f"Skipped index {idx}")

        # IMPORTANT: slow down to avoid throttling
        time.sleep(0.15)

# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    print("Downloading TRAIN images...")
    download_images("../data/raw/train.csv", "../images/train")

    print("Downloading TEST images...")
    download_images("../data/raw/test.csv", "../images/test")
