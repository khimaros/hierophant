#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <map>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

[[noreturn]] inline void fatal(const char * fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    fprintf(stderr, "error: ");
    vfprintf(stderr, fmt, ap);
    fprintf(stderr, "\n");
    va_end(ap);
    exit(1);
}

// llama.cpp's short names for the model options every command takes
static const std::map<std::string, std::string> ARG_ALIASES = {
    { "-hf",  "--hf-repo"  },
    { "-hfr", "--hf-repo"  },
    { "-hff", "--hf-file"  },
    { "-hft", "--hf-token" },
};
static const char * ARG_OFFLINE  = "--offline";
static const char * ARG_PROGRESS = "--progress"; // every command takes it, as the scripts pass it to any

// the bar scripts/progress.py draws, in the same format, so a parent script can tell
// redraws (ending in a carriage return) from finished lines
static const int    PROGRESS_WIDTH  = 24;
static const double PROGRESS_REDRAW = 0.2; // seconds between redraws
static const char * PROGRESS_CLEAR  = "\x1b[K";

// command line of the form `--key value` and bare `--flag`, after the subcommand.
// a repeated key keeps every value; str() returns the last.
struct args {
    std::map<std::string, std::vector<std::string>> kv;

    args(int argc, char ** argv, const std::vector<std::string> & flags) {
        for (int i = 0; i < argc; i++) {
            std::string k = argv[i];
            if (k.rfind("-", 0) != 0) {
                fatal("unexpected argument '%s'", k.c_str());
            }
            if (auto alias = ARG_ALIASES.find(k); alias != ARG_ALIASES.end()) {
                k = alias->second;
            }
            bool is_flag = k == ARG_OFFLINE || k == ARG_PROGRESS;
            for (const auto & f : flags) {
                is_flag |= f == k;
            }
            if (is_flag) {
                kv[k].push_back("1");
            } else if (i + 1 < argc) {
                kv[k].push_back(argv[++i]);
            } else {
                fatal("missing value for '%s'", k.c_str());
            }
        }
    }

    bool has(const std::string & k) const { return kv.count(k) > 0; }

    std::string str(const std::string & k, const std::string & def = "") const {
        auto it = kv.find(k);
        return it == kv.end() ? def : it->second.back();
    }

    std::vector<std::string> all(const std::string & k) const {
        auto it = kv.find(k);
        return it == kv.end() ? std::vector<std::string>{} : it->second;
    }

    std::string need(const std::string & k) const {
        if (!has(k)) {
            fatal("missing required argument '%s'", k.c_str());
        }
        return str(k);
    }

    int i32(const std::string & k, int def) const { return has(k) ? std::stoi(str(k)) : def; }

    double f64(const std::string & k, double def) const { return has(k) ? std::stod(str(k)) : def; }
};

inline std::vector<std::string> split(const std::string & s, char sep) {
    std::vector<std::string> out;
    size_t                   start = 0;
    while (start <= s.size()) {
        size_t end = s.find(sep, start);
        if (end == std::string::npos) {
            end = s.size();
        }
        if (end > start) {
            out.push_back(s.substr(start, end - start));
        }
        start = end + 1;
    }
    return out;
}

inline bool ends_with(const std::string & s, const std::string & suffix) {
    return s.size() >= suffix.size() && s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

inline std::string basename_of(const std::string & path) {
    const size_t slash = path.find_last_of('/');
    return slash == std::string::npos ? path : path.substr(slash + 1);
}

inline std::string clock_of(double seconds) {
    const long s = (long) seconds, m = s / 60;
    char       buf[32];
    if (m >= 60) {
        snprintf(buf, sizeof(buf), "%ld:%02ld:%02ld", m / 60, m % 60, s % 60);
    } else {
        snprintf(buf, sizeof(buf), "%ld:%02ld", m, s % 60);
    }
    return buf;
}

// units finished so far: with --progress a bar redrawn in place and ended by a line
// of its own; without, "label: done/total unit" every `every` units (none for 0).
struct progress {
    using clock = std::chrono::steady_clock;
    std::string       label, unit;
    size_t            total, every;
    bool              bar;
    clock::time_point start = clock::now(), drawn = start;

    progress(const args & a, std::string label, size_t total, size_t every = 0, std::string unit = "") :
        label(std::move(label)), unit(std::move(unit)), total(total), every(every), bar(a.has(ARG_PROGRESS)) {
        if (bar) {
            draw(0, "\r");
        }
    }

    void draw(size_t done, const char * end) {
        drawn                = clock::now();
        const double elapsed = std::chrono::duration<double>(drawn - start).count();
        const double frac    = total ? (double) done / total : 1.0;
        const int    fill    = (int) std::lround(PROGRESS_WIDTH * frac);
        std::string  eta     = done > 0 && done < total ? " eta " + clock_of(elapsed / done * (total - done)) : "";
        fprintf(stderr, "%s [%s%s] %zu/%zu %.0f%% %s%s%s%s", label.c_str(), std::string(fill, '#').c_str(),
                std::string(PROGRESS_WIDTH - fill, '-').c_str(), done, total, frac * 100, clock_of(elapsed).c_str(),
                eta.c_str(), PROGRESS_CLEAR, end);
        fflush(stderr);
    }

    void update(size_t done) {
        if (!bar) {
            if (every && (done % every == 0 || done == total)) {
                fprintf(stderr, "%s: %zu/%zu %s\n", label.c_str(), done, total, unit.c_str());
            }
        } else if (done == total) {
            draw(done, "\n");
        } else if (std::chrono::duration<double>(clock::now() - drawn).count() >= PROGRESS_REDRAW) {
            draw(done, "\r");
        }
    }

    // a line of its own, over the bar when one is drawn
    std::string eol() const { return bar ? std::string(PROGRESS_CLEAR) + "\n" : "\n"; }
};

inline int default_threads() {
    unsigned n = std::thread::hardware_concurrency();
    return n == 0 ? 4 : (int) n;
}

// run fn(begin, end, worker) over [0, n) split into contiguous chunks.
inline void parallel_for(int64_t n, int n_threads, const std::function<void(int64_t, int64_t, int)> & fn) {
    n_threads = (int) std::max<int64_t>(1, std::min<int64_t>(n_threads, n));
    if (n_threads == 1) {
        fn(0, n, 0);
        return;
    }
    std::vector<std::thread> workers;
    const int64_t            chunk = (n + n_threads - 1) / n_threads;
    for (int w = 0; w < n_threads; w++) {
        const int64_t b = w * chunk;
        const int64_t e = std::min(n, b + chunk);
        if (b < e) {
            workers.emplace_back(fn, b, e, w);
        }
    }
    for (auto & t : workers) {
        t.join();
    }
}
