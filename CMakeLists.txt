cmake_minimum_required(VERSION 3.21)
project(hierophant C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# the llama.cpp build every adapter is validated against. override the source
# with -DFETCHCONTENT_SOURCE_DIR_LLAMA_CPP=<checkout> to reuse a local tree.
set(LLAMA_CPP_REPO   "https://github.com/ggml-org/llama.cpp")
set(LLAMA_CPP_COMMIT "8e5a58df87ab59dfceb5b961ae26691e39f30c24")

set(LLAMA_BUILD_COMMON   ON  CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS    ON  CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER   ON  CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_APP      OFF CACHE BOOL "" FORCE)
# https for -hf downloads, from the boringssl version llama.cpp pins rather than
# whatever openssl the host has
set(LLAMA_OPENSSL        OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_BORINGSSL ON CACHE BOOL "" FORCE)
set(LLAMA_USE_PREBUILT_UI OFF CACHE BOOL "" FORCE)

include(FetchContent)
FetchContent_Declare(llama_cpp GIT_REPOSITORY ${LLAMA_CPP_REPO} GIT_TAG ${LLAMA_CPP_COMMIT})
FetchContent_MakeAvailable(llama_cpp)

execute_process(
    COMMAND git -C ${llama_cpp_SOURCE_DIR} rev-parse HEAD
    OUTPUT_VARIABLE LLAMA_CPP_BUILT_COMMIT OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
if(NOT LLAMA_CPP_BUILT_COMMIT STREQUAL LLAMA_CPP_COMMIT)
    message(WARNING "llama.cpp source is at '${LLAMA_CPP_BUILT_COMMIT}', pinned commit is ${LLAMA_CPP_COMMIT}")
endif()

add_executable(hierophant
    src/main.cpp
    src/hf_cache.cpp
    src/model_file.cpp
    src/sites.cpp
    src/directions.cpp
    src/capture.cpp
    src/adapter.cpp)
target_link_libraries(hierophant PRIVATE llama llama-common ggml)
target_compile_definitions(hierophant PRIVATE LLAMA_CPP_COMMIT="${LLAMA_CPP_BUILT_COMMIT}")
