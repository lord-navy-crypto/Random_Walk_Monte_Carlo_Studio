# Contributing

Contributions are welcome through GitHub issues and pull requests.

1. Create a focused branch.
2. Keep the interface English-only.
3. Preserve the Visualization → Key results → Complete data order.
4. Add or update tests for numerical changes.
5. Run `python -m compileall -q rw_mc_studio app.py` and `python -m pytest -q`.
6. Explain numerical assumptions and uncertainty semantics in the pull request.

Do not commit virtual environments, caches, generated figures, private data, or large simulation outputs.
