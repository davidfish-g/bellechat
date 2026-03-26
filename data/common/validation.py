"""
Data quality validation and anachronism detection for bellechat.
"""

import os
import re
import random
import logging

logger = logging.getLogger(__name__)

# Terms that strongly indicate post-1914 knowledge.
# If any of these appear in pretraining data, it's likely contaminated.
ANACHRONISM_TERMS = [
    # Wars & geopolitics
    "world war", "first world war", "second world war", "wwi", "wwii",
    "treaty of versailles", "league of nations", "united nations",
    "nazi", "fascist", "fascism", "gestapo", "holocaust",
    "soviet union", "soviet russia", "bolshevik revolution",
    "cold war", "iron curtain", "berlin wall",
    "korean war", "vietnam war",
    # Technology
    "television", "radio broadcast", "broadcasting",
    "computer", "software", "internet", "website", "email",
    "transistor", "semiconductor", "microchip",
    "nuclear power", "nuclear weapon", "atomic bomb", "hydrogen bomb",
    "nuclear reactor", "nuclear fission", "nuclear fusion",
    "jet engine", "jet aircraft", "supersonic",
    "satellite", "space station", "moon landing",
    "laser", "radar",
    # Science & medicine
    "antibiotic", "penicillin", "insulin therapy",
    "quantum mechanics", "quantum theory", "heisenberg",
    "general relativity", "black hole", "big bang",
    "dna molecule", "double helix", "genetic code",
    # People (born after ~1900 or famous only post-1914)
    "hitler", "mussolini", "stalin", "lenin",
    "fdr", "franklin roosevelt",
    # Concepts
    "world war i", "world war ii", "interwar",
    "great depression",
    "decolonization", "decolonisation",
]

# Compile patterns for efficient matching
_ANACHRONISM_PATTERNS = [
    re.compile(r"\b" + re.escape(term) + r"\b", re.IGNORECASE)
    for term in ANACHRONISM_TERMS
]


def scan_for_anachronisms(text: str) -> list[str]:
    """
    Scan text for post-1914 terms. Returns list of matched terms.
    Empty list means no anachronisms detected.
    """
    found = []
    for pattern, term in zip(_ANACHRONISM_PATTERNS, ANACHRONISM_TERMS):
        if pattern.search(text):
            found.append(term)
    return found


def estimate_tokens(text: str) -> int:
    """Rough token count estimate (~3.5 chars per token for English text)."""
    return int(len(text) / 3.5)


def corpus_summary(
    source_dir: str,
    source_name: str,
    sample_count: int = 5,
    sample_chars: int = 500,
) -> dict:
    """
    Print summary statistics for a collected corpus directory.
    Returns stats dict with doc_count, total_chars, estimated_tokens.
    """
    if not os.path.isdir(source_dir):
        logger.warning(f"Directory not found: {source_dir}")
        return {"doc_count": 0, "total_chars": 0, "estimated_tokens": 0}

    files = [f for f in os.listdir(source_dir) if f.endswith(".txt")]
    total_chars = 0
    doc_lengths = []
    anachronism_hits = {}

    for fname in files:
        path = os.path.join(source_dir, fname)
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()
        total_chars += len(text)
        doc_lengths.append(len(text))
        # Spot-check for anachronisms
        found = scan_for_anachronisms(text)
        if found:
            anachronism_hits[fname] = found

    estimated_tokens = estimate_tokens(str(total_chars))
    stats = {
        "doc_count": len(files),
        "total_chars": total_chars,
        "estimated_tokens": int(total_chars / 3.5),
    }

    print(f"\n{'='*60}")
    print(f"  Corpus Summary: {source_name}")
    print(f"{'='*60}")
    print(f"  Documents:        {stats['doc_count']:,}")
    print(f"  Total characters: {stats['total_chars']:,}")
    print(f"  Estimated tokens: {stats['estimated_tokens']:,}")
    if doc_lengths:
        doc_lengths.sort()
        print(f"  Doc length min:   {doc_lengths[0]:,}")
        print(f"  Doc length median: {doc_lengths[len(doc_lengths)//2]:,}")
        print(f"  Doc length max:   {doc_lengths[-1]:,}")

    if anachronism_hits:
        print(f"\n  WARNING: {len(anachronism_hits)} documents contain anachronistic terms:")
        for fname, terms in list(anachronism_hits.items())[:10]:
            print(f"    {fname}: {', '.join(terms)}")
        if len(anachronism_hits) > 10:
            print(f"    ... and {len(anachronism_hits) - 10} more")
    else:
        print(f"\n  No anachronisms detected.")

    # Print random samples
    if files and sample_count > 0:
        sample_files = random.sample(files, min(sample_count, len(files)))
        print(f"\n  Random samples ({sample_count} docs, first {sample_chars} chars each):")
        for fname in sample_files:
            path = os.path.join(source_dir, fname)
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                excerpt = f.read(sample_chars)
            print(f"\n  --- {fname} ---")
            print(f"  {excerpt}...")

    print(f"{'='*60}\n")
    return stats


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python -m data.common.validation <source_dir> <source_name>")
        sys.exit(1)
    corpus_summary(sys.argv[1], sys.argv[2])
