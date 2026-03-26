"""
Parse the Project Gutenberg RDF catalog to build a manifest of pre-1914 English works.

The RDF catalog is a tarball containing one RDF/XML file per ebook. We parse each
to extract metadata and filter for works verifiably published before 1914.

Date filtering strategy:
  - Primary: author death year < 1914 → all works qualify
  - Secondary: author death year >= 1914 but MARC 260 field or subject headings
    indicate the work was published before 1914
  - Fallback: if no date can be determined, exclude the text

Usage:
    python -m data.gutenberg.gutenberg_rdf [--output manifest.csv] [--cache-dir DIR]
"""

import os
import re
import csv
import tarfile
import logging
import argparse
import urllib.request
from xml.etree import ElementTree as ET

logger = logging.getLogger(__name__)

CATALOG_URL = "https://www.gutenberg.org/cache/epub/feeds/rdf-files.tar.bz2"

# XML namespaces used in Gutenberg RDF files
NS = {
    "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    "dcterms": "http://purl.org/dc/terms/",
    "pgterms": "http://www.gutenberg.org/2009/pgterms/",
    "dcam": "http://purl.org/dc/dcam/",
    "cc": "http://web.resource.org/cc/",
}


def download_catalog(cache_dir: str) -> str:
    """Download the RDF catalog tarball if not already cached."""
    catalog_path = os.path.join(cache_dir, "rdf-files.tar.bz2")
    if os.path.exists(catalog_path):
        logger.info(f"Catalog already cached at {catalog_path}")
        return catalog_path

    os.makedirs(cache_dir, exist_ok=True)
    logger.info(f"Downloading RDF catalog from {CATALOG_URL}...")
    urllib.request.urlretrieve(CATALOG_URL, catalog_path)
    logger.info(f"Catalog saved to {catalog_path}")
    return catalog_path


def _extract_text(el, default=""):
    """Safely extract text from an XML element."""
    if el is not None and el.text:
        return el.text.strip()
    return default


def _extract_year(text: str) -> int | None:
    """Try to extract a 4-digit year from a string."""
    match = re.search(r"\b(\d{4})\b", text)
    if match:
        year = int(match.group(1))
        if 1000 <= year <= 2100:
            return year
    return None


def parse_rdf_entry(xml_bytes: bytes) -> dict | None:
    """
    Parse a single Gutenberg RDF/XML file and extract metadata.
    Returns a dict with ebook info, or None if parsing fails.
    """
    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError:
        return None

    ebook = root.find("pgterms:ebook", NS)
    if ebook is None:
        return None

    # Extract ebook ID from rdf:about attribute
    about = ebook.get(f"{{{NS['rdf']}}}about", "")
    ebook_id_match = re.search(r"/(\d+)$", about)
    if not ebook_id_match:
        return None
    ebook_id = int(ebook_id_match.group(1))

    # Title
    title = _extract_text(ebook.find("dcterms:title", NS), "")

    # Language
    lang_el = ebook.find("dcterms:language/rdf:Description/rdf:value", NS)
    language = _extract_text(lang_el, "")

    # Author info
    creator = ebook.find("dcterms:creator/pgterms:agent", NS)
    author_name = ""
    author_birth = None
    author_death = None
    if creator is not None:
        author_name = _extract_text(creator.find("pgterms:name", NS), "")
        birth_el = creator.find("pgterms:birthdate", NS)
        death_el = creator.find("pgterms:deathdate", NS)
        if birth_el is not None:
            author_birth = _extract_year(_extract_text(birth_el))
        if death_el is not None:
            author_death = _extract_year(_extract_text(death_el))

    # Subjects (useful for date heuristics)
    subjects = []
    for subj in ebook.findall("dcterms:subject", NS):
        desc = subj.find("rdf:Description/rdf:value", NS)
        if desc is not None and desc.text:
            subjects.append(desc.text.strip())

    # MARC fields — look for publication date hints
    # dcterms:description sometimes has original publication info
    description = _extract_text(ebook.find("dcterms:description", NS), "")

    # Check for text format availability
    has_text = False
    for fmt in ebook.findall("dcterms:hasFormat/pgterms:file", NS):
        about_url = fmt.get(f"{{{NS['rdf']}}}about", "")
        if about_url.endswith(".txt") or "text/plain" in about_url:
            has_text = True
            break
        # Also check format descriptions
        for fmt_desc in fmt.findall("dcterms:format/rdf:Description/rdf:value", NS):
            if fmt_desc.text and "text/plain" in fmt_desc.text:
                has_text = True
                break

    return {
        "ebook_id": ebook_id,
        "title": title,
        "author": author_name,
        "author_birth": author_birth,
        "author_death": author_death,
        "language": language,
        "subjects": subjects,
        "description": description,
        "has_text": has_text,
    }


def is_pre_1914(entry: dict) -> tuple[bool, str]:
    """
    Determine if a work was published before 1914.
    Returns (qualifies, reason).
    """
    # Primary: author died before 1914 → all their works qualify
    if entry["author_death"] is not None and entry["author_death"] < 1914:
        return True, f"author died {entry['author_death']}"

    # If author died 1914+ or death unknown, look for publication date clues
    # Check subjects for date hints (e.g., "English fiction -- 19th century")
    for subj in entry["subjects"]:
        subj_lower = subj.lower()
        # Subjects mentioning centuries before 20th are safe
        for century in ["17th century", "18th century", "19th century",
                        "to 1500", "to 1800", "to 1900"]:
            if century in subj_lower:
                return True, f"subject indicates pre-1914: {subj}"

    # Check description for publication year
    if entry["description"]:
        year = _extract_year(entry["description"])
        if year is not None and year < 1914:
            return True, f"description mentions year {year}"

    # Author born before 1850 with no death date → very likely pre-1914
    if entry["author_birth"] is not None and entry["author_birth"] < 1850:
        if entry["author_death"] is None:
            return True, f"author born {entry['author_birth']}, likely pre-1914"

    # No author at all (anonymous) — could be ancient text, but risky
    # Skip these for safety unless subjects give a strong signal
    if not entry["author"]:
        for subj in entry["subjects"]:
            subj_lower = subj.lower()
            if any(kw in subj_lower for kw in ["ancient", "classical", "medieval",
                                                 "bible", "mythology", "folklore"]):
                return True, f"anonymous but subject indicates historical: {subj}"
        return False, "anonymous, no date signal"

    # Author with death date >= 1914 but no publication date evidence
    if entry["author_death"] is not None and entry["author_death"] >= 1914:
        return False, f"author died {entry['author_death']}, no pre-1914 evidence"

    # No date information at all — exclude for safety
    return False, "no date information available"


def build_manifest(catalog_path: str, output_path: str, language_filter: str = "en"):
    """
    Parse the full RDF catalog and write a CSV manifest of qualifying pre-1914 works.
    """
    included = 0
    excluded = 0
    no_text = 0
    parse_errors = 0

    with open(output_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=[
            "ebook_id", "title", "author", "author_death", "year_reason",
            "language", "url",
        ])
        writer.writeheader()

        logger.info("Parsing RDF catalog (this may take a few minutes)...")
        with tarfile.open(catalog_path, "r:bz2") as tar:
            for member in tar:
                if not member.name.endswith(".rdf"):
                    continue

                f = tar.extractfile(member)
                if f is None:
                    continue

                entry = parse_rdf_entry(f.read())
                if entry is None:
                    parse_errors += 1
                    continue

                # Language filter
                if language_filter and entry["language"] != language_filter:
                    excluded += 1
                    continue

                # Must have a text format
                if not entry["has_text"]:
                    no_text += 1
                    continue

                # Date filter
                qualifies, reason = is_pre_1914(entry)
                if not qualifies:
                    excluded += 1
                    continue

                writer.writerow({
                    "ebook_id": entry["ebook_id"],
                    "title": entry["title"],
                    "author": entry["author"],
                    "author_death": entry["author_death"] or "",
                    "year_reason": reason,
                    "language": entry["language"],
                    "url": f"https://www.gutenberg.org/files/{entry['ebook_id']}/{entry['ebook_id']}-0.txt",
                })
                included += 1

                if included % 5000 == 0:
                    logger.info(f"  ... {included} works included so far")

    logger.info(f"Manifest complete: {included} included, {excluded} excluded, "
                f"{no_text} no text format, {parse_errors} parse errors")
    logger.info(f"Written to {output_path}")
    return included


def main():
    parser = argparse.ArgumentParser(description="Build pre-1914 Gutenberg manifest from RDF catalog")
    parser.add_argument("--cache-dir", default=os.path.expanduser("~/.cache/bellechat"),
                        help="Directory to cache the RDF catalog download")
    parser.add_argument("--output", default=None,
                        help="Output CSV path (default: <cache-dir>/gutenberg_manifest.csv)")
    parser.add_argument("--language", default="en",
                        help="Language filter (default: en, empty string for all)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    output = args.output or os.path.join(args.cache_dir, "gutenberg", "manifest.csv")
    catalog_path = download_catalog(os.path.join(args.cache_dir, "gutenberg"))
    count = build_manifest(catalog_path, output, language_filter=args.language)
    print(f"\nDone. {count} pre-1914 English works in {output}")


if __name__ == "__main__":
    main()
