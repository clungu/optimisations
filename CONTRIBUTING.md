# Contributing

This is a notebook-first project. The numbered notebooks in the repository root
are the source of truth for `optimisations/`. The generated Python modules and
`optimisations/_modidx.py` should be committed, but edited through their
notebooks rather than directly. `index.ipynb` is the source of `README.md`.

## Set up a development environment

Use Python 3.10 or newer:

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[dev]'
nbdev-install-hooks
```

The editable installation makes changes to exported modules available in the
environment. The `dev` extra installs nbdev, Jupyter tooling, and the dependencies
needed to generate the demo preview.

## Add or change functionality

1. Find the numbered notebook that owns the relevant module. For a new module,
   add the next numbered notebook and give it a `#| default_exp module_name`
   directive.
2. Add implementation cells marked `#| export`. Use `#| exporti` for code that
   belongs in the generated module but should not be part of its public
   `__all__`. Keep explanatory Markdown and examples close to the code.
3. Add assertions in notebook cells and regression tests in `tests/`. Keep
   notebook tests reproducible from a fresh kernel: do not depend on manually
   executed cells or saved outputs.
4. Export and test the change:

   ```sh
   nbdev-export
   python -m unittest discover -s tests
   nbdev-test
   ```

   `make test` runs these steps together. Review the changes to both the
   notebook and its generated module before committing them.

If you need to edit generated Python temporarily in an IDE, `nbdev-update` can
propagate changes back to notebooks. Review the resulting notebook diff; the
notebook remains the source of truth.

## Build the README and documentation

Install [Quarto](https://quarto.org/docs/get-started/) separately, then run:

```sh
make docs
```

This runs `nbdev-docs` and `nbdev-readme`. The site is built into the ignored
`_docs/` directory. The tracked `docs/` directory is a historical Jekyll
snapshot, **not** the current nbdev3 site. Building `_docs/` does not publish it;
site deployment must be configured separately. Use `nbdev-preview` to inspect
the site locally.

Edit `index.ipynb`, not the generated `README.md`. The README's GIF preview
comes from `README-example.gif`; run `./demo.sh` when the example changes and
commit the updated GIF. The script also writes an interactive HTML version to
`output/README-example.html` for local viewing. GitHub READMEs do not execute
its JavaScript, so the GIF is the inline preview.

Keep expensive examples from running during every documentation build. Use
`#| eval: false` for demonstrations that should be shown but not executed.
`#| include: false` hides a cell from rendered documentation; it does **not**
serve as a substitute for disabling execution.

## Build and install the package

For a local wheel:

```sh
make dist
```

This exports the notebooks and builds a wheel in `dist/`. It does not build a
source distribution. Inspect the filename in `dist/` and install the intended
version, preferably into a fresh virtual environment:

```sh
python -m pip install dist/optimisations-<version>-py3-none-any.whl
```

Replace `<version>` with the actual filename; do not paste the angle brackets
literally. To install directly from a checkout without development tools, use
`python -m pip install .`. For notebook development, use
`python -m pip install -e '.[dev]'` instead.

MP4 rendering requires the system `ffmpeg` executable. Prefer `output='js'`
for portable examples and tests.

## Publish a release to PyPI

Publishing is a maintainer task. You need permission to publish the
`optimisations` project on PyPI and an appropriately scoped PyPI API token or
trusted-publishing configuration.

1. Choose a version that has not previously been published. Update **both**
   `[project].version` in `pyproject.toml` and `__version__` in
   `optimisations/__init__.py` to the same value. Reinstall the editable project
   if you changed its package metadata.
2. Export, run the tests, regenerate documentation, review the generated
   changes, and commit the release sources:

   ```sh
   python -m pip install -e '.[dev]'
   make test
   make docs
   ```

   CI checks that exported modules and the README match their notebook sources.
3. Build a source distribution and wheel in a **fresh, version-specific**
   directory, then check both artifacts:

   ```sh
   python -m pip install --upgrade build twine
   VERSION=0.0.2  # Replace with the version in pyproject.toml
   python -m build --outdir "dist/release-$VERSION"
   python -m twine check "dist/release-$VERSION"/*
   ```

   Inspect the artifact filenames and contents. In particular, verify that
   notebook sources are present in the source distribution and that the wheel
   contains the exported modules. Check how the README image renders on PyPI:
   its current relative GIF path is designed for GitHub and may not resolve on
   a PyPI project page.
4. After checking the artifacts and confirming that the release commit is on
   the intended branch, upload only that version's artifacts:

   ```sh
   python -m twine upload "dist/release-$VERSION"/*
   ```

   Never put a PyPI token in a notebook, source file, command committed to Git,
   or documentation. PyPI does not allow replacing an uploaded artifact with
   the same filename; a correction needs a new version.

## nbdev3 tips

- nbdev configuration and package metadata live in `pyproject.toml`, not the
  legacy `settings.ini` or `setup.py`.
- Run `nbdev-export` after notebook edits. Commit both notebook changes and
  generated package changes; do not hand-maintain two implementations.
- Use `nbdev-test` to execute notebook tests and the standard-library
  `unittest` suite for focused package regressions.
- Keep saved notebook outputs small and free of errors. The installed
  `nbdev-install-hooks` hooks help clean notebooks before committing.
- Keep GitHub-facing demonstrations lightweight. GitHub renders images and
  GIFs in Markdown but will not run an embedded JavaScript animation.