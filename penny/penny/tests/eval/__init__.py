"""Live-model eval suite — the faithful replacement for the old
scripts/prompt_validation harness.

These drive the REAL agents (chat + collector) against a running Ollama on
synthetic seeds and assert behavioural contracts.  They are slow and stochastic,
so they're excluded from ``make check`` (marked ``eval``) and run via
``make eval``.  See docs/self-improvement-loop.md.
"""
