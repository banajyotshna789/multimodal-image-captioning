import os
import json
import gzip
import re
import pandas as pd

DATA_DIR = "data"
LISTINGS_DIR = os.path.join(DATA_DIR, "listings", "metadata")
IMAGE_META_PATH = os.path.join(DATA_DIR, "images", "metadata", "images.csv.gz")
IMAGE_ROOT = os.path.join(DATA_DIR, "images", "small")
OUTPUT_FILE = "clean_data.csv"
MAX_SAMPLES = 5000

# Keep only footwear-related product types
ALLOWED_PRODUCT_TYPES = {
    "SHOES",
    "SANDAL",
    "SANDALS",
    "BOOT",
    "BOOTS",
    "SNEAKER",
    "SNEAKERS",
    "LOAFER",
    "LOAFERS",
    "HEEL",
    "HEELS",
    "PUMP",
    "PUMPS",
    "SLIPPER",
    "SLIPPERS",
    "FLAT",
    "FLATS",
    "MULE",
    "MULES",
    "TRAINER",
    "TRAINERS",
    "OXFORD",
    "BROGUE",
}

ALLOWED_COLORS = [
    "black", "brown", "blue", "red", "white", "grey", "gray", "tan", "gold",
    "silver", "green", "pink", "purple", "beige", "navy", "orange", "yellow",
    "olive", "maroon", "mushroom", "multicolor", "multicolour", "white", "black"
]

ALLOWED_MATERIALS = [
    "leather", "suede", "synthetic", "cotton", "canvas", "polyester",
    "mesh", "rubber", "fabric", "denim", "wool", "velvet", "lycra", "pu"
]

ALLOWED_STYLE_WORDS = [
    "formal", "casual", "sport", "running", "loafer", "sandals", "sneakers",
    "boots", "heels", "slip on", "slip-on", "trainer", "trainers", "mules",
    "pump", "pumps", "slippers", "brogue", "oxford", "flat", "flats",
    "fashion", "dress", "office"
]

def extract_text(field):
    if not isinstance(field, list):
        return ""

    for item in field:
        if isinstance(item, dict):
            lang = str(item.get("language_tag", ""))
            val = item.get("value", "")
            if lang.startswith("en") and val:
                return str(val).strip()

    for item in field:
        if isinstance(item, dict):
            val = item.get("value", "")
            if val:
                return str(val).strip()

    return ""

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def dedupe_words(text):
    words = text.split()
    out = []
    for w in words:
        if not out or out[-1] != w:
            out.append(w)
    return " ".join(out)

def build_image_lookup():
    if not os.path.exists(IMAGE_META_PATH):
        raise FileNotFoundError(f"Missing image metadata file: {IMAGE_META_PATH}")

    img_df = pd.read_csv(IMAGE_META_PATH)
    if "image_id" not in img_df.columns or "path" not in img_df.columns:
        raise ValueError("images.csv.gz must contain 'image_id' and 'path' columns")

    img_df["full_image_path"] = img_df["path"].apply(lambda p: os.path.join(IMAGE_ROOT, p))
    return dict(zip(img_df["image_id"], img_df["full_image_path"]))

def detect_gender(text):
    if "women" in text or "womens" in text or "woman" in text or "female" in text:
        return "women"
    if "men" in text or "mens" in text or "man" in text or "male" in text:
        return "men"
    if "girl" in text or "girls" in text:
        return "girls"
    if "boy" in text or "boys" in text:
        return "boys"
    return ""

def detect_color(text):
    for c in ALLOWED_COLORS:
        if c in text:
            return c
    return ""

def detect_material(text, fabric_text=""):
    combined = f"{text} {clean_text(fabric_text)}"
    for m in ALLOWED_MATERIALS:
        if m in combined:
            return m
    return ""

def detect_style(text, bullets=""):
    combined = f"{text} {bullets}"
    for s in ALLOWED_STYLE_WORDS:
        if s in combined:
            return s.replace("-", " ")
    return ""

def build_caption(item_name, brand="", color="", fabric="", bullet_text="", product_type=""):
    item_name = clean_text(item_name)
    brand = clean_text(brand)
    color = clean_text(color)
    fabric = clean_text(fabric)
    bullet_text = clean_text(bullet_text)
    product_type = clean_text(product_type)

    text = f"{item_name} {brand} {color} {fabric} {bullet_text}"

    gender = detect_gender(text)
    detected_color = detect_color(text) or color
    material = detect_material(text, fabric)
    style = detect_style(text, bullet_text)

    parts = []
    if gender:
        parts.append(gender)
    if detected_color:
        parts.append(detected_color)
    if material:
        parts.append(material)
    if style:
        parts.append(style)

    # Use a cleaner footwear noun
    if product_type:
        if "shoe" in product_type or "sneaker" in product_type or "trainer" in product_type or "loafer" in product_type:
            noun = "shoes"
        elif "sandal" in product_type:
            noun = "sandals"
        elif "boot" in product_type:
            noun = "boots"
        elif "heel" in product_type or "pump" in product_type:
            noun = "heels"
        elif "slipper" in product_type:
            noun = "slippers"
        elif "flat" in product_type:
            noun = "flats"
        elif "mule" in product_type:
            noun = "mules"
        else:
            noun = product_type
        parts.append(noun)

    caption = " ".join([p for p in parts if p]).strip()
    caption = clean_text(caption)
    caption = dedupe_words(caption)
    caption = re.sub(r"\s+", " ", caption).strip()
    return caption

def main():
    if not os.path.isdir(LISTINGS_DIR):
        raise FileNotFoundError(f"Listings directory not found: {LISTINGS_DIR}")

    img_map = build_image_lookup()

    rows = []
    meta_files = sorted(f for f in os.listdir(LISTINGS_DIR) if f.endswith(".json.gz"))

    if len(meta_files) == 0:
        raise RuntimeError(f"No listing metadata shards found in {LISTINGS_DIR}")

    for fname in meta_files:
        shard_path = os.path.join(LISTINGS_DIR, fname)

        with gzip.open(shard_path, "rt", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line)

                    product_type = extract_text(data.get("product_type", []))
                    product_type_upper = product_type.upper().strip()

                    # strict footwear-only filter
                    if product_type_upper not in ALLOWED_PRODUCT_TYPES:
                        continue

                    item_name = extract_text(data.get("item_name", []))
                    brand = extract_text(data.get("brand", []))
                    color = extract_text(data.get("color", []))
                    fabric = extract_text(data.get("fabric_type", []))
                    bullet_text = extract_text(data.get("bullet_point", []))

                    caption = build_caption(
                        item_name=item_name,
                        brand=brand,
                        color=color,
                        fabric=fabric,
                        bullet_text=bullet_text,
                        product_type=product_type_upper.lower()
                    )

                    if len(caption.split()) < 2:
                        continue

                    main_image_id = data.get("main_image_id", "")
                    if not main_image_id:
                        continue

                    image_path = img_map.get(main_image_id, "")
                    if not image_path:
                        continue

                    if not os.path.exists(image_path):
                        continue

                    rows.append({
                        "image_path": image_path,
                        "caption": caption,
                        "item_id": data.get("item_id", ""),
                        "product_type": product_type_upper
                    })

                    if len(rows) >= MAX_SAMPLES:
                        break

                except Exception:
                    continue

        if len(rows) >= MAX_SAMPLES:
            break

    df = pd.DataFrame(rows)

    if len(df) == 0:
        raise RuntimeError("No samples were collected. Check the image mapping and strict filter.")

    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Collected samples: {len(df)}")
    print(f"Saved {OUTPUT_FILE}")
    print(df.head())

if __name__ == "__main__":
    main()
