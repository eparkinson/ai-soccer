Poetry migration notes

This repository was converted to use Poetry for local development.

Quick start (local):

1. Install Poetry if you don't have it: https://python-poetry.org/docs/#installation

2. Create and activate the virtual environment and install deps:

```bash
poetry install
poetry shell
```

3. Run demos inside the Poetry shell:

```bash
python demo_game.py
```

Notes:
- `pyproject.toml` was created from `requirements.txt` with the same top-level packages.
- If you use a specific version of a dependency, update `pyproject.toml` before running `poetry install`.
- To export a `requirements.txt` for deployment: `poetry export -f requirements.txt --output requirements.txt --without-hashes`

