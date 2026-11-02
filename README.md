# bellechat

A 1.4 billion parameter LLM trained from scratch on text published before 1914. It has never heard of the world wars, nuclear physics, antibiotics or the Soviet Union, and it answers as a knowledgeable, articulate person from 1913.

Built as a fork of Karpathy's [nanochat](https://github.com/karpathy/nanochat). Weights are on [Hugging Face](https://huggingface.co/david-fish/bellechat).

## Model

- **Parameters:** 1.4B (24 layers, 1536 dim, 12 heads)
- **Pretraining:** 11.8B tokens of pre-1914 text
- **Fine-tuning:** 6,200 synthetic conversations for conversational ability
- **Training time:** ~3.5 hours on 8×H100 with FP8

It's a small model, so expect some incoherence, and the fine-tuning is likely overfit.

## Data

| Source | Files | Tokens | Notes |
|---|---|---|---|
| Project Gutenberg | 20,729 | 2.68B | Filtered by author death date |
| 1911 Encyclopaedia Britannica | 36,221 | 0.07B | Complete 11th edition, 5× upweighted |
| Chronicling America | 1.3M | 6.7B | Library of Congress newspaper OCR, filtered by date and OCR quality |
| Internet Archive | 21,935 | 2.3B | Filtered by date metadata |

- Newspaper pages with under 80% valid words are dropped.
- Modern Project Gutenberg and Internet Archive headers and footers are stripped.
- The corpus was scanned for post-1914 terms; contamination is under 0.01%.
- Fine-tuning conversations were generated with explicit constraints against modern knowledge.

## Run it locally

Requires Python 3.10+ and Git.

```bash
git clone https://github.com/davidfish-g/bellechat.git
cd bellechat
python run.py
```

This installs dependencies, downloads the model (~4 GB), and opens the chat at http://localhost:8000 once the model has loaded. To uninstall, delete the `bellechat` folder and `~/.cache/bellechat`.
