# contributing

- `make` builds, `make test-e2e` runs the end-to-end tests, and `make
  precommit` runs ruff and clang-format. run `make format` to apply the C++
  style, which is llama.cpp's.
- toolchain versions are pinned in `mise.toml`. the llama.cpp commit is pinned
  in `CMakeLists.txt`; bump it deliberately and rerun the E2E tests, because
  adapter layouts and the eval callback's tensor names are llama.cpp internals.
- C++ follows llama.cpp's idiom: `snake_case`, `static const` named constants
  at the top of each file, `fatal()` for user-facing errors.
- python uses the standard library only. a new user-facing tool gets a
  command in the `hierophant` entrypoint. tests drive that and the built
  binaries as subprocesses and check results through `scripts/gguf_lite.py`, never
  through the C++ code under test.
- prefer an end-to-end test against a real GGUF over a unit test. the tiny
  models in `scripts/fetch_models.py` cover a dense model with tied
  embeddings and a mixture-of-experts model with untied embeddings.
- `scripts/hierophant_search.py` mirrors the curve formula in
  `src/adapter.cpp`; change both together. `tests/test_search.py` checks
  that they agree.
- prompt sets are not committed: several carry no license or a
  non-commercial one. `make datasets` fetches them and checks each against
  the SHA-256 in `data/manifest.json`; commit only the manifest, and bump a
  pin with `hierophant data fetch --update` deliberately, as with llama.cpp.
- bug fixes start with a failing test.
- output, comments and docs are lowercase ASCII, with acronyms in capitals
  (KL, LoRA, GGUF, HF, JSON, SHA-256). flags, file names, keys and code keep
  their own spelling.
