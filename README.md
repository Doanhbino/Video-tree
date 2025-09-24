# Adaptive Tree-based Video Representation for LLM Reasoning on Long Videos

This project builds an **adaptive hierarchical (tree) representation** of a video and connects it to an LLM for **summarization** and **question answering** on long videos.

## What it does
- Extracts frames from a video at a configurable rate
- Generates CLIP embeddings for each frame (offline via `transformers`)
- Builds a multi-level tree by **merging adjacent segments** with high similarity
- (Optional) Generates lightweight captions for segments via BLIP (offline) or an LLM
- Ranks relevant segments given a user query (CLIP text–image joint space) and crafts an LLM answer grounded in retrieved segments
- Provides a **Streamlit web demo** to upload a video, build the tree, browse it, ask questions, and get summaries

## Quickstart

### 1) Python env
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

> If you have a CUDA GPU, install a CUDA-enabled PyTorch build before the rest:
> https://pytorch.org/get-started/locally/

### 2) (Optional) OpenAI API for LLM answers
Create an `.env` file from the template and set your key:
```bash
cp .env.example .env
# edit .env and add OPENAI_API_KEY=...
```

### 3) Run the app
```bash
streamlit run app.py
```

Upload a video (`.mp4`, `.mov`, etc.). The app will extract frames, build the hierarchy, and let you ask questions.

### 4) CLI usage (optional)
```bash
python run_local.py --video path/to/video.mp4 --fps 0.5 --similarity 0.88
```

This will create a JSON tree and embeddings under `data/`.

## Project layout
```
adaptive-video-tree-llm/
├─ app.py                      # Streamlit demo
├─ run_local.py                # CLI entry point
├─ requirements.txt
├─ .env.example
├─ src/
│  ├─ config.py
│  ├─ utils/
│  │  ├─ visualize.py
│  ├─ video/
│  │  └─ preprocess.py
│  ├─ features/
│  │  ├─ clip_extractor.py
│  │  └─ captioner.py
│  ├─ tree/
│  │  └─ builder.py
│  └─ llm/
│     └─ query.py
├─ data/                       # outputs (frames, embeddings, tree.json)
├─ tests/
│  └─ test_smoke.py
└─ README.md
```

## Notes
- Default CLIP model: `openai/clip-vit-base-patch32` (works on CPU; faster on GPU)
- If BLIP is too heavy, leave `CAPTIONING_ENABLED=False` in `src/config.py` and the app will skip offline captioning
- The **tree algorithm** is intentionally simple and *deterministic*: it merges adjacent segments bottom-up until similarity falls below a threshold, then repeats at higher levels. You can swap in a more advanced hierarchical clustering if you like.

## License
MIT
