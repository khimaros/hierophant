BUILD_DIR ?= build
JOBS ?= $(shell nproc)
# reuse a local llama.cpp checkout instead of fetching the pinned commit
LLAMA_CPP_DIR ?=
CMAKE_ARGS ?=
# where make sync sends this checkout
HOST ?=
REMOTE ?= src/hierophant

CMAKE_FLAGS := -G Ninja -DCMAKE_BUILD_TYPE=Release $(CMAKE_ARGS)
ifneq ($(LLAMA_CPP_DIR),)
CMAKE_FLAGS += -DFETCHCONTENT_SOURCE_DIR_LLAMA_CPP=$(abspath $(LLAMA_CPP_DIR))
endif

PYTHON := python3 -I
# the last set fetch_datasets.py writes; present means all of them are
DATASETS := data/harmful-lat.txt

.PHONY: build configure test test-e2e precommit format datasets models clean sync

build: configure
	cmake --build $(BUILD_DIR) -j $(JOBS) --target hierophant-core llama-server llama-cli llama-perplexity

configure: $(BUILD_DIR)/build.ninja

$(BUILD_DIR)/build.ninja: CMakeLists.txt
	cmake -S . -B $(BUILD_DIR) $(CMAKE_FLAGS)

test: test-e2e

test-e2e: build models $(DATASETS)
	$(PYTHON) -m unittest discover -s tests -v

datasets:
	$(PYTHON) hierophant data fetch data --more

$(DATASETS):
	$(PYTHON) hierophant data fetch data --more

models:
	$(PYTHON) hierophant data models models

precommit: build
	ruff check hierophant scripts tests
	clang-format --dry-run --Werror src/*.cpp src/*.h

format:
	clang-format -i src/*.cpp src/*.h

clean:
	rm -rf $(BUILD_DIR)

# configs travel both ways, the newer file winning; everything else is pushed.
# pull first: the checkout there is overwritten, and a build there writes its config
sync:
	@test -n "$(HOST)" || { echo "usage: make sync HOST=kairos [REMOTE=src/hierophant]"; exit 1; }
	rsync -rtuv $(HOST):$(REMOTE)/configs/ configs/
	rsync -rlptDv --exclude-from=.gitignore --exclude=.git/ --exclude=configs/ --delete ./ $(HOST):$(REMOTE)/
	rsync -rtuv configs/ $(HOST):$(REMOTE)/configs/
