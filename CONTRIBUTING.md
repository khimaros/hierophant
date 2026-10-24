# contributing

- `make` builds, `make test-e2e` runs the end-to-end tests, and `make
  precommit` runs ruff and clang-format. run `make format` to apply the c++
  style, which is llama.cpp's.
- toolchain versions are pinned in `mise.toml`. the llama.cpp commit is pinned
  in `CMakeLists.txt`; bump it deliberately and rerun the e2e tests, because
  adapter layouts and the eval callback's tensor names are llama.cpp internals.
- c++ follows llama.cpp's idiom: `snake_case`, `static const` named constants
  at the top of each file, `fatal()` for user-facing errors.
- python uses the standard library only. tests drive the built binaries as
  subprocesses and check results through `scripts/gguf_lite.py`, never
  through the c++ code under test.
- prefer an end-to-end test against a real gguf over a unit test. the tiny
  models in `scripts/fetch_models.py` cover a dense model with tied
  embeddings and a mixture-of-experts model with untied embeddings.
- `scripts/hierophant_search.py` mirrors the curve formula in
  `src/adapter.cpp`; change both together. `tests/test_search.py` checks
  that they agree.
- prompt sets are not committed: several carry no license or a
  non-commercial one. `make datasets` fetches them and checks each against
  the sha256 in `data/manifest.json`; commit only the manifest, and bump a
  pin with `fetch_datasets.py --update` deliberately, as with llama.cpp.
- bug fixes start with a failing test.
- output, comments and docs are lowercase ascii.
