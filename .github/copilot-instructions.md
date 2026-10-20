## Project Rules

- Follow `.specify/memory/constitution.md` and the current feature plan when one exists.
- Use Python for modeling and data workflows. Manage dependencies only with uv:
	`uv add` / `uv add --dev`, `uv sync --locked`, and `uv run`. Commit `pyproject.toml`
	and `uv.lock`. Never use `pip`, `python -m pip`, or `uv pip`, including in notebooks,
	CI, documentation examples, or Azure Machine Learning environment builds.
- Target Azure Machine Learning v2 jobs for reproducible training and evaluation;
	remote environments must synchronize the uv lockfile without pip-based setup.
- Preserve NASA C-MAPSS provenance and citation; prevent engine-level and temporal
	leakage. Test affected data/model behavior and track per-subset evaluation results.
- Keep credentials, datasets, and model binaries out of Git. Obtain authorization
	before cloud provisioning or paid runs, and report unrun checks explicitly.
<!-- SPECKIT START -->
For additional context about technologies to be used, project structure,
shell commands, and other important information, read the current plan
<!-- SPECKIT END -->
