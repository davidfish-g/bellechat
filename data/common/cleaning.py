"""
Shared text cleaning utilities for bellechat data collection.
"""

import re
import unicodedata


def normalize_unicode(text: str) -> str:
    """NFC normalization and common encoding fixes."""
    text = unicodedata.normalize("NFC", text)
    # Fix common mojibake patterns
    replacements = {
        "\u00e2\u0080\u0099": "\u2019",  # right single quote
        "\u00e2\u0080\u009c": "\u201c",  # left double quote
        "\u00e2\u0080\u009d": "\u201d",  # right double quote
        "\u00e2\u0080\u0094": "\u2014",  # em dash
        "\u00e2\u0080\u0093": "\u2013",  # en dash
        "\ufeff": "",  # BOM
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def collapse_whitespace(text: str) -> str:
    """Collapse 3+ consecutive newlines to 2, normalize trailing whitespace on lines."""
    # Normalize line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Strip trailing whitespace on each line
    text = re.sub(r"[ \t]+$", "", text, flags=re.MULTILINE)
    # Collapse 3+ newlines to 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Strip leading/trailing whitespace from entire document
    text = text.strip()
    return text


# Patterns for Gutenberg header/footer markers
_GUTENBERG_START = re.compile(
    r"\*\*\*\s*START OF (?:THE |THIS )?PROJECT GUTENBERG.*?\*\*\*",
    re.IGNORECASE,
)
_GUTENBERG_END = re.compile(
    r"\*\*\*\s*END OF (?:THE |THIS )?PROJECT GUTENBERG.*?\*\*\*",
    re.IGNORECASE,
)


def remove_gutenberg_headers(text: str) -> str:
    """Strip Project Gutenberg header and footer boilerplate."""
    # Find start marker — take everything after it
    match = _GUTENBERG_START.search(text)
    if match:
        text = text[match.end():]

    # Find end marker — take everything before it
    match = _GUTENBERG_END.search(text)
    if match:
        text = text[:match.start()]

    return text.strip()


def remove_illustration_tags(text: str) -> str:
    """Remove [Illustration: ...] tags common in Gutenberg texts."""
    return re.sub(r"\[Illustration[^\]]*\]", "", text, flags=re.IGNORECASE)


def estimate_non_ascii_ratio(text: str) -> float:
    """Return fraction of characters that are non-ASCII (indicator of bad OCR)."""
    if not text:
        return 0.0
    non_ascii = sum(1 for c in text if ord(c) > 127)
    return non_ascii / len(text)


def estimate_ocr_quality(text: str, word_set: set[str]) -> float:
    """
    Estimate OCR quality by checking what fraction of words are in a dictionary.
    Returns a ratio between 0.0 (all garbage) and 1.0 (all valid words).
    """
    words = re.findall(r"[a-zA-Z]+", text.lower())
    if not words:
        return 0.0
    valid = sum(1 for w in words if w in word_set)
    return valid / len(words)


def remove_modern_boilerplate(text: str) -> str:
    """Remove modern boilerplate that leaked into pre-1914 texts. Line-based for speed."""
    lines = text.split("\n")
    cleaned = []
    skip = False
    for line in lines:
        lower = line.lower()
        # Start skipping at Gutenberg license / small print blocks
        if "project gutenberg" in lower and ("license" in lower or "ebook" in lower or "e-book" in lower or "electronic" in lower):
            skip = True
            continue
        if "full project gutenberg" in lower:
            skip = True
            continue
        # Internet Archive digitization notices
        if "digitized by" in lower and "internet archive" in lower:
            continue
        if "internet archive" in lower and ("produced from" in lower or "generated from" in lower):
            continue
        # URLs
        if "www.gutenberg.org" in lower or "gutenberg.org" in lower:
            continue
        if "archive.org" in lower:
            continue
        if "http://" in lower or "https://" in lower:
            line = re.sub(r"https?://\S+", "", line)
        if "www." in lower:
            line = re.sub(r"www\.\S+", "", line)
        if not skip:
            cleaned.append(line)
    return "\n".join(cleaned).strip()


def is_quality_text(
    text: str,
    min_length: int = 1000,
    max_non_ascii_ratio: float = 0.05,
) -> bool:
    """Combined quality gate for cleaned text."""
    if len(text) < min_length:
        return False
    if estimate_non_ascii_ratio(text) > max_non_ascii_ratio:
        return False
    return True
