# bellechat

A conversational LLM trained from scratch on text published before 1914. It knows only what the world knew during the Belle Époque — no world wars, no nuclear physics, no antibiotics, no aviation beyond the Wright brothers, no radio broadcasting, no Soviet Union. You can talk to it and it will answer as a knowledgeable, articulate person from 1913 would.

Built as a fork of Karpathy's [nanochat](https://github.com/karpathy/nanochat).

## Model

- **Parameters:** 1.4B (d24 transformer, 24 layers, 1536 dim, 12 heads)
- **Training data:** 13.3B tokens from pre-1914 text (Project Gutenberg, 1911 Encyclopaedia Britannica, Chronicling America newspapers, Internet Archive)
- **SFT:** 6,200 synthetic conversations for persona and conversational ability
- **Training time:** ~3.5 hours pretraining + SFT on 8xH100 SXM with FP8

Model weights: [david-fish/bellechat](https://huggingface.co/david-fish/bellechat)

## Quick start

```bash
git clone https://github.com/davidfish-g/bellechat.git
cd bellechat
pip install uv && uv sync --extra cpu
source .venv/bin/activate
pip install huggingface-hub
```

Download the model:

```bash
python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    'david-fish/bellechat',
    local_dir='~/.cache/bellechat',
    allow_patterns=['chatsft_checkpoints/**', 'tokenizer/**'],
)
"
```

Run it:

```bash
# Web UI (ChatGPT-style)
python -m scripts.chat_web

# Or chat in the terminal
python -m scripts.chat_cli
```

## Data sources

| Source | Files | Tokens | Description |
|--------|-------|--------|-------------|
| Project Gutenberg | 20,729 | 2.68B | Pre-1914 books filtered by author death date |
| 1911 Britannica | 36,221 | 0.07B | Complete 11th edition (5x upweighted in training) |
| Chronicling America | 1.3M | 6.7B | LOC newspaper OCR, 80% quality filtered |
| Internet Archive | 21,935 | 2.3B | Pre-1914 texts via IA search API |

## Data Quality Improvements

- **Date filtering:** Gutenberg books filtered by author death date. Newspaper pages filtered by publication date in file path. IA texts filtered by date metadata.
- **OCR quality:** Newspaper pages below 80% word validity ratio are rejected.
- **Boilerplate removal:** Modern Project Gutenberg and Internet Archive headers/footers are stripped at shard time.
- **Anachronism detection:** Corpus scanned for post-1914 terms. Contamination rate: <0.01%.
- **SFT conversations:** Generated with explicit constraints against modern knowledge leakage.

## Acknowledgments

Built on [nanochat](https://github.com/karpathy/nanochat) by Andrej Karpathy.
