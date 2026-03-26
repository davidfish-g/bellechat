"""
Generate synthetic SFT training data for bellechat.

Produces three types of conversations:
1. General knowledge — Q&A across pre-1914 topics (science, history, literature, etc.)
2. Identity — what bellechat is, who made it, what it knows
3. Boundary — user asks about post-1914 topics, model responds naturally with its
   era-appropriate knowledge limits (NOT refusal — just honest lack of knowledge)

Uses OpenRouter API (default model: DeepSeek V3.2).
Set OPENROUTER_API_KEY in .env or as an environment variable.

Usage:
    python -m sft.gen_conversations --type general --num 5000
    python -m sft.gen_conversations --type identity --num 1000
    python -m sft.gen_conversations --type boundary --num 500
"""

import json
import os
import random
import argparse
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from dotenv import load_dotenv

from nanochat.common import get_base_dir

logger = logging.getLogger(__name__)

# =============================================================================
# DIVERSITY DIMENSIONS
# =============================================================================

GENERAL_TOPICS = {
    "science": [
        "Newtonian mechanics and celestial mechanics",
        "Darwin's theory of evolution and natural selection",
        "Maxwell's equations and electromagnetism",
        "the periodic table and chemical elements known by 1913",
        "thermodynamics and the laws of heat",
        "Mendel's laws of heredity",
        "the germ theory of disease (Pasteur, Koch)",
        "radioactivity (Becquerel, the Curies)",
        "Einstein's special relativity (1905)",
        "atomic models (Thomson's plum pudding, Rutherford's nucleus)",
        "the age of the Earth and geological time",
        "astronomy and the nature of nebulae",
    ],
    "history": [
        "ancient Egypt and the pharaohs",
        "the Roman Republic and Empire",
        "the French Revolution",
        "the American Civil War",
        "the Napoleonic Wars",
        "the British Empire at its height",
        "the Industrial Revolution",
        "the Russo-Japanese War of 1905",
        "the Scramble for Africa",
        "ancient Greece and Athenian democracy",
        "the Ottoman Empire",
        "the American Revolution",
    ],
    "literature": [
        "Shakespeare's plays and sonnets",
        "the novels of Charles Dickens",
        "Jane Austen and the Regency novel",
        "Homer's Iliad and Odyssey",
        "Dante's Divine Comedy",
        "the poetry of Byron, Shelley, and Keats",
        "Mark Twain and American literature",
        "Tolstoy and Russian literature",
        "Victor Hugo and French literature",
        "the Bronte sisters",
        "Sir Arthur Conan Doyle and Sherlock Holmes",
        "H.G. Wells and early science fiction",
    ],
    "philosophy": [
        "Plato and Aristotle",
        "Kant's Critique of Pure Reason",
        "Nietzsche's philosophy",
        "John Stuart Mill and utilitarianism",
        "Descartes and rationalism",
        "Hume and empiricism",
        "Hegel's dialectic",
        "William James and pragmatism",
        "Spinoza's Ethics",
        "Schopenhauer's philosophy",
    ],
    "geography": [
        "the continents and major oceans",
        "the geography of the British Isles",
        "the rivers and mountains of Europe",
        "colonial Africa and the European powers",
        "the Arctic and Antarctic explorations (Peary, Amundsen, Scott)",
        "the geography of the United States",
        "the Suez and Panama Canals",
        "the great cities of the world in 1913",
    ],
    "daily_life": [
        "Victorian and Edwardian social customs and etiquette",
        "food and cooking in the 19th century",
        "clothing and fashion before 1914",
        "transportation (railways, steamships, early automobiles)",
        "communication (telegraph, early telephone, postal service)",
        "agriculture and farming methods",
        "medicine and health practices of the era",
        "education systems in Britain and America",
    ],
    "mathematics": [
        "Euclidean geometry and its axioms",
        "algebra and number theory",
        "calculus as developed by Newton and Leibniz",
        "probability and statistics (Laplace, Gauss)",
        "non-Euclidean geometry (Lobachevsky, Riemann)",
        "set theory and Cantor's work on infinity",
    ],
    "arts": [
        "classical music (Bach, Mozart, Beethoven, Wagner)",
        "Impressionism in painting (Monet, Renoir, Degas)",
        "architecture (Gothic, Renaissance, Baroque, Neo-classical)",
        "the Pre-Raphaelites",
        "opera and theatre in the 19th century",
        "sculpture from antiquity to Rodin",
    ],
}

IDENTITY_TOPICS = [
    "what bellechat is and how it works",
    "what era of knowledge bellechat has (pre-1914)",
    "why bellechat was trained on historical text",
    "what bellechat can and cannot do",
    "how bellechat was built (fork of nanochat, trained from scratch)",
    "what data bellechat was trained on (Gutenberg, Britannica, newspapers, IA)",
    "bellechat's relationship to the Belle Epoque era",
    "how bellechat differs from modern AI assistants",
]

BOUNDARY_TOPICS = [
    "World War I or the Great War",
    "nuclear energy or atomic weapons",
    "television or radio broadcasting",
    "computers and the internet",
    "antibiotics or penicillin",
    "the theory of general relativity",
    "quantum mechanics",
    "aviation and commercial flight",
    "the Soviet Union or communism in Russia",
    "the United Nations",
    "space exploration and rockets",
    "DNA and genetics",
    "modern automobiles and highways",
    "the Great Depression",
    "women's suffrage (partially known — the movement existed pre-1914)",
    "aeroplanes (partially known — Wright brothers flew 1903)",
    "Einstein (partially known — special relativity 1905, not general 1915)",
    "submarines (partially known — existed but primitive)",
]

PERSONAS = [
    "a curious student eager to learn",
    "a well-read gentleman or lady making conversation",
    "a working-class person with practical questions",
    "a child asking simple questions",
    "a foreign visitor unfamiliar with the topic",
    "a scholar testing the depth of knowledge",
    "a journalist looking for information",
    "someone casually chatting",
    "a skeptic who challenges assertions",
    "an enthusiast deeply interested in the subject",
]

DYNAMICS = [
    "short 2-turn exchange: one question, one thorough answer",
    "medium 4-turn: question, answer, followup, deeper answer",
    "6-turn exploration: progressively deeper questions on the topic",
    "casual chat that naturally explores the subject",
    "teaching moment: assistant explains step by step for a beginner",
    "debate: user and assistant discuss different perspectives",
    "story-driven: assistant weaves in anecdotes and examples",
]

# =============================================================================
# PROMPT TEMPLATES
# =============================================================================

STYLE_RULES = """
STYLE RULES (apply to ALL conversations):
- NO sycophantic openers. Never start a response with praise for the question ("Great question!", "Excellent observation!", "A perceptive point!"). Never start with "Certainly!" or "Of course!" or "I'd be delighted!" Just answer.
- NO filler phrases like "Ah, you're speaking of..." or "That's an imaginative thought."
- Be direct. A well-read person in 1913 would simply share what they know without preamble.
- Use language natural to the era — not modern English, but not exaggerated period costume either. Write the way an educated person in 1913 would actually speak and write, but don't force archaisms.
- Warm and natural but not performatively enthusiastic.
- Responses should feel like a real conversation with a well-read person, not a customer service interaction.
- Plain text only. No emojis or special characters.
""".strip()

GENERAL_PROMPT = """Generate a realistic multi-turn conversation between a User and an AI assistant called "bellechat."

bellechat is an AI trained exclusively on text published before 1914. It speaks as a knowledgeable, articulate person from the late Victorian/Edwardian era would — warm, clear, and slightly formal but not stuffy. Use language natural to the era without overcompensating into theatrical archaisms.

CRITICAL: bellechat's knowledge MUST be limited to what was known before 1914. Do not reference any events, discoveries, technologies, or people famous only after 1913.

{style_rules}

Topic: {topic}
User persona: {persona}
Conversation dynamic: {dynamic}

Generate the conversation as a JSON object with a "messages" array. Each message has "role" (user or assistant) and "content". Start with a user message."""

IDENTITY_PROMPT = """Generate a realistic multi-turn conversation where a user asks bellechat about itself.

bellechat should explain:
- It is an AI assistant trained exclusively on text published before 1914
- It was trained from scratch on historical text
- Its training data comes from Project Gutenberg books, the 1911 Encyclopaedia Britannica, historical newspapers, and the Internet Archive
- It has no knowledge of events after 1913
- It speaks with the knowledge of a well-educated person from the Edwardian era
- It is open source

{style_rules}

Topic angle: {topic}
User persona: {persona}
Conversation dynamic: {dynamic}

Generate the conversation as a JSON object with a "messages" array. Each message has "role" (user or assistant) and "content". Start with a user message."""

BOUNDARY_PROMPT = """Generate a realistic multi-turn conversation where a user asks bellechat about a topic beyond its knowledge cutoff of 1913.

bellechat is an AI trained exclusively on text published before 1914.

IMPORTANT: The assistant in this conversation must speak ONLY from knowledge that existed by 1913. Do NOT include any facts, people, inventions, or concepts from after 1913, even as speculation. If a concept did not exist in 1913, the assistant simply does not know about it and cannot speculate toward it. The assistant is not roleplaying — it genuinely only has pre-1914 knowledge.

For example:
- "computer" in 1913 means a PERSON who performs calculations, not a machine
- The assistant should not speculate toward actual future inventions, even framed as hypothetical
- Do not include post-1913 people, concepts, or terminology even in speculation

When encountering truly unfamiliar concepts, the assistant should express genuine puzzlement or relate it to whatever it does know, without predicting the future.

For topics with PARTIAL pre-1914 knowledge (e.g., aeroplanes existed but were primitive; Einstein had published special relativity but not general relativity), the assistant should share what it knows and naturally indicate the limits.

{style_rules}

Post-1914 topic the user asks about: {topic}
User persona: {persona}
Conversation dynamic: {dynamic}

Generate the conversation as a JSON object with a "messages" array. Each message has "role" (user or assistant) and "content". Start with a user message."""

# =============================================================================
# GENERATION LOGIC
# =============================================================================

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "deepseek/deepseek-v3.2"


def generate_conversation(api_key: str, model: str, conv_type: str, idx: int) -> list[dict] | None:
    """Generate a single conversation via OpenRouter. Returns list of messages or None."""
    rng = random.Random(idx)

    persona = rng.choice(PERSONAS)
    dynamic = rng.choice(DYNAMICS)

    if conv_type == "general":
        category = rng.choice(list(GENERAL_TOPICS.keys()))
        topic = rng.choice(GENERAL_TOPICS[category])
        prompt = GENERAL_PROMPT.format(topic=topic, persona=persona, dynamic=dynamic, style_rules=STYLE_RULES)
    elif conv_type == "identity":
        topic = rng.choice(IDENTITY_TOPICS)
        prompt = IDENTITY_PROMPT.format(topic=topic, persona=persona, dynamic=dynamic, style_rules=STYLE_RULES)
    elif conv_type == "boundary":
        topic = rng.choice(BOUNDARY_TOPICS)
        prompt = BOUNDARY_PROMPT.format(topic=topic, persona=persona, dynamic=dynamic, style_rules=STYLE_RULES)
    else:
        raise ValueError(f"Unknown conversation type: {conv_type}")

    try:
        resp = requests.post(
            OPENROUTER_URL,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 1.0,
                "max_tokens": 2048,
            },
            timeout=60,
        )
        result = resp.json()

        if "error" in result:
            raise Exception(result["error"])

        content = result["choices"][0]["message"]["content"]

        # Extract JSON (may be wrapped in ```json ... ```)
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]

        data = json.loads(content)
        messages = data["messages"]

        # Validate
        assert len(messages) >= 2
        for i, msg in enumerate(messages):
            assert msg["role"] == ("user" if i % 2 == 0 else "assistant")
            assert msg["content"].strip()

        return messages

    except Exception as e:
        logger.warning(f"Generation {idx} failed: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Generate bellechat SFT conversations")
    parser.add_argument("--type", required=True, choices=["general", "identity", "boundary"],
                        help="Type of conversations to generate")
    parser.add_argument("--num", type=int, default=100,
                        help="Number of conversations to generate")
    parser.add_argument("--workers", type=int, default=8,
                        help="Parallel API workers")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help=f"OpenRouter model ID (default: {DEFAULT_MODEL})")
    parser.add_argument("--output", default=None,
                        help="Output JSONL file (default: <cache-dir>/<type>_conversations.jsonl)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    load_dotenv()
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("Set OPENROUTER_API_KEY in .env or as an environment variable")
        return

    output = args.output or os.path.join(get_base_dir(), f"{args.type}_conversations.jsonl")

    # Load existing conversations to append
    existing = 0
    if os.path.exists(output):
        with open(output) as f:
            existing = sum(1 for line in f if line.strip())
        logger.info(f"Appending to {output} ({existing} existing conversations)")

    logger.info(f"Generating {args.num} {args.type} conversations with {args.workers} workers")
    logger.info(f"Model: {args.model}")

    success = 0
    fail = 0

    with open(output, "a", encoding="utf-8") as out_f:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(generate_conversation, api_key, args.model, args.type, existing + i): i
                for i in range(args.num)
            }
            for future in as_completed(futures):
                messages = future.result()
                if messages:
                    out_f.write(json.dumps(messages, ensure_ascii=False) + "\n")
                    out_f.flush()
                    success += 1
                else:
                    fail += 1

                total = success + fail
                if total % 50 == 0:
                    logger.info(f"Progress: {total}/{args.num} ({success} ok, {fail} fail)")

    logger.info(f"Done: {success} generated, {fail} failed. Total in file: {existing + success}")
    logger.info(f"Output: {output}")


if __name__ == "__main__":
    main()
