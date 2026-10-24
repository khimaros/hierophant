BUILD_DIR ?= build
JOBS ?= $(shell nproc)
# reuse a local llama.cpp checkout instead of fetching the pinned commit
LLAMA_CPP_DIR ?=
CMAKE_ARGS ?=

CMAKE_FLAGS := -G Ninja -DCMAKE_BUILD_TYPE=Release $(CMAKE_ARGS)
ifneq ($(LLAMA_CPP_DIR),)
CMAKE_FLAGS += -DFETCHCONTENT_SOURCE_DIR_LLAMA_CPP=$(abspath $(LLAMA_CPP_DIR))
endif

PYTHON := python3 -I
# the last set fetch_datasets.py writes; present means all of them are
DATASETS := data/harmful-lat.txt

.PHONY: build configure test test-e2e precommit format datasets models clean

build: configure
	cmake --build $(BUILD_DIR) -j $(JOBS) --target hierophant llama-server llama-perplexity

configure: $(BUILD_DIR)/build.ninja

$(BUILD_DIR)/build.ninja: CMakeLists.txt
	cmake -S . -B $(BUILD_DIR) $(CMAKE_FLAGS)

test: test-e2e

test-e2e: build models $(DATASETS)
	$(PYTHON) -m unittest discover -s tests -v

datasets:
	$(PYTHON) scripts/fetch_datasets.py data --more

$(DATASETS):
	$(PYTHON) scripts/fetch_datasets.py data --more

models:
	$(PYTHON) scripts/fetch_models.py models

precommit: build
	ruff check scripts tests
	clang-format --dry-run --Werror src/*.cpp src/*.h

format:
	clang-format -i src/*.cpp src/*.h

clean:
	rm -rf $(BUILD_DIR)
