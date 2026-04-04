"""
Download Maimonides texts for training.
Run this script locally: python data/download_data.py

Sources:
- Guide for the Perplexed (Friedländer translation) from Project Gutenberg
- Additional texts from Sefaria API (Creative Commons)
"""

import urllib.request
import json
import os
import re
import time

DATA_DIR = os.path.dirname(os.path.abspath(__file__))


def clean_gutenberg_text(text: str) -> str:
    """Remove Project Gutenberg headers/footers and clean up the text."""
    # Find start of actual content (after the Gutenberg header)
    start_markers = ["*** START OF THE PROJECT GUTENBERG", "*** START OF THIS PROJECT GUTENBERG"]
    end_markers = ["*** END OF THE PROJECT GUTENBERG", "*** END OF THIS PROJECT GUTENBERG"]

    for marker in start_markers:
        idx = text.find(marker)
        if idx != -1:
            text = text[idx + len(marker):]
            # Skip past the rest of that line
            text = text[text.find("\n") + 1:]
            break

    for marker in end_markers:
        idx = text.find(marker)
        if idx != -1:
            text = text[:idx]
            break

    # Clean up excessive whitespace but keep paragraph breaks
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = text.strip()
    return text


def download_guide_for_perplexed() -> str:
    """Download Guide for the Perplexed from Project Gutenberg."""
    print("Downloading Guide for the Perplexed from Project Gutenberg...")
    url = "https://www.gutenberg.org/cache/epub/73584/pg73584.txt"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "TinyRambam/1.0"})
        with urllib.request.urlopen(req) as response:
            text = response.read().decode("utf-8")
        text = clean_gutenberg_text(text)
        print(f"  Got {len(text):,} characters")
        return text
    except Exception as e:
        print(f"  Failed: {e}")
        # Try alternate URL
        try:
            alt_url = "https://www.gutenberg.org/files/73584/73584-0.txt"
            req = urllib.request.Request(alt_url, headers={"User-Agent": "TinyRambam/1.0"})
            with urllib.request.urlopen(req) as response:
                text = response.read().decode("utf-8")
            text = clean_gutenberg_text(text)
            print(f"  Got {len(text):,} characters (alternate URL)")
            return text
        except Exception as e2:
            print(f"  Alternate also failed: {e2}")
            return ""


def download_sefaria_text(ref: str, pause: float = 0.5) -> str:
    """Download a text from Sefaria's free API."""
    encoded_ref = urllib.parse.quote(ref)
    url = f"https://www.sefaria.org/api/texts/{encoded_ref}?lang=en"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "TinyRambam/1.0"})
        with urllib.request.urlopen(req) as response:
            data = json.loads(response.read().decode("utf-8"))

        # Extract English text - Sefaria returns nested lists
        def extract_text(obj):
            if isinstance(obj, str):
                # Strip HTML tags
                clean = re.sub(r'<[^>]+>', '', obj)
                return clean.strip()
            elif isinstance(obj, list):
                parts = [extract_text(item) for item in obj]
                return "\n".join(p for p in parts if p)
            return ""

        text = extract_text(data.get("text", ""))
        time.sleep(pause)  # Be polite to the API
        return text
    except Exception as e:
        print(f"  Failed to get {ref}: {e}")
        return ""


def download_mishneh_torah_selections() -> str:
    """Download key philosophical sections of Mishneh Torah from Sefaria."""
    print("Downloading Mishneh Torah selections from Sefaria...")

    # These are the most philosophical/theological sections
    sections = [
        "Mishneh Torah, Foundations of the Torah",
        "Mishneh Torah, Human Dispositions",
        "Mishneh Torah, Torah Study",
        "Mishneh Torah, Repentance",
        "Mishneh Torah, Foreign Worship and Customs of the Nations",
    ]

    all_text = []
    for section in sections:
        print(f"  Downloading: {section}")
        text = download_sefaria_text(section)
        if text:
            all_text.append(f"\n\n--- {section} ---\n\n{text}")
            print(f"    Got {len(text):,} characters")
        else:
            print(f"    Skipped (no text returned)")

    combined = "\n".join(all_text)
    print(f"  Total Mishneh Torah: {len(combined):,} characters")
    return combined


def download_guide_sefaria() -> str:
    """Download Guide for the Perplexed from Sefaria as backup/supplement."""
    print("Downloading Guide for the Perplexed from Sefaria...")
    parts = [
        "Guide for the Perplexed, Introduction",
        "Guide for the Perplexed, Part 1",
        "Guide for the Perplexed, Part 2",
        "Guide for the Perplexed, Part 3",
    ]

    all_text = []
    for part in parts:
        print(f"  Downloading: {part}")
        text = download_sefaria_text(part, pause=1.0)
        if text:
            all_text.append(text)
            print(f"    Got {len(text):,} characters")

    combined = "\n\n".join(all_text)
    print(f"  Total Guide (Sefaria): {len(combined):,} characters")
    return combined


def main():
    import urllib.parse

    all_texts = []

    # 1. Guide for the Perplexed from Gutenberg (primary source)
    guide_gutenberg = download_guide_for_perplexed()
    if guide_gutenberg:
        all_texts.append(guide_gutenberg)

    # 2. Guide from Sefaria (supplement / backup if Gutenberg fails)
    if not guide_gutenberg:
        guide_sefaria = download_guide_sefaria()
        if guide_sefaria:
            all_texts.append(guide_sefaria)

    # 3. Mishneh Torah philosophical sections
    mishneh = download_mishneh_torah_selections()
    if mishneh:
        all_texts.append(mishneh)

    # Combine everything
    corpus = "\n\n".join(all_texts)

    # Save raw corpus
    output_path = os.path.join(DATA_DIR, "maimonides_corpus.txt")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(corpus)

    print(f"\nDone! Saved {len(corpus):,} characters to {output_path}")
    print(f"That's roughly {len(corpus.split()):,} words")


if __name__ == "__main__":
    main()