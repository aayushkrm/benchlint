# Limitations

Status: stub. Written during integration. Must cover at least:

- No paraphrase detection (no LLM or embeddings).
- No per-item membership inference (unreliable on closed models).
- Shortcut results depend on the chosen views and subsample size.
- Results are CPU-only, measured on an Intel i7-4770HQ laptop, macOS x86_64; CI runs on Linux.
