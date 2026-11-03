# phylokit

Phylogenetics toolkit for the PyData Ecosystem

**This package is experimental DO NOT USE!**

## Development

phylokit uses [uv](https://docs.astral.sh/uv/) for dependency management and
supports Python 3.12, 3.13 and 3.14. To set up a development environment and
run the tests:

```bash
uv sync
uv run pytest
```

Linting and formatting use [ruff](https://docs.astral.sh/ruff/), run through
[prek](https://github.com/j178/prek) as configured in `prek.toml`:

```bash
uv run prek install
uv run prek run --all-files
```

To update the locked dependencies, run `uv lock --upgrade` and commit the
resulting `uv.lock`.
