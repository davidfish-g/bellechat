# bellechat

A conversational LLM trained from scratch on text published before 1914—no world wars, no nuclear physics, no antibiotics, no Soviet Union, etc. You can talk to it and it will answer as a knowledgeable, articulate person from 1913 would.

Built as a fork of Karpathy's [nanochat](https://github.com/karpathy/nanochat).

## Model

- **Parameters:** 1.4B (d24 transformer, 24 layers, 1536 dim, 12 heads)
- **Training data:** 13.3B tokens from pre-1914 text (Project Gutenberg, 1911 Encyclopaedia Britannica, Chronicling America newspapers, Internet Archive)
- **SFT:** 6,200 synthetic conversations for persona and conversational ability
- **Training time:** ~3.5 hours pretraining + SFT on 8xH100 SXM with FP8

Model weights: [david-fish/bellechat](https://huggingface.co/david-fish/bellechat)

**Note:** This is a 1.4B parameter model so expect some incoherence. Also, the SFT is likely overfit. I plan to scale significantly and improve SFT when I have more resources.

## Quick start

**Requirements:** Python 3.10+ and Git.

```bash
git clone https://github.com/davidfish-g/bellechat.git
cd bellechat
./run.sh
```

`run.sh` installs dependencies, downloads the model (~2 GB), and starts the server. Then open http://localhost:8000 in your browser.

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