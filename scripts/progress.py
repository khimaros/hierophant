"""a one-line progress bar on stderr, redrawn in place.

each redraw ends in a carriage return and its last line in a newline, so a
parent reading the output (hierophant_pipeline's Run.sh) can tell the two
apart: it draws redraws in place and keeps only finished lines in its logs.
"""

import re
import sys
import threading
import time

WIDTH = 24
REDRAW_EVERY = 0.2  # seconds between redraws
CLEAR = "\x1b[K"  # erase to the end of the line, past a longer earlier redraw


def clock(seconds):
    m, s = divmod(int(seconds), 60)
    return f"{m // 60}:{m % 60:02d}:{s:02d}" if m >= 60 else f"{m}:{s:02d}"


COUNT = re.compile(r"(?P<label>.*) \[[#-]*\] (?P<done>\d+)/(?P<total>\d+) ")  # a rendered bar's own count


def render(label, done, total, elapsed):
    """'label [#####-------] 12/40 30% 0:14 eta 0:33'. done may be fractional, a unit
    under way; the count shows only finished ones."""
    frac = min(done / total, 1.0) if total else 1.0
    fill = round(WIDTH * frac)
    eta = f" eta {clock(elapsed / done * (total - done))}" if 0 < done < total else ""
    return f"{label} [{'#' * fill}{'-' * (WIDTH - fill)}] {int(done)}/{total} {frac:.0%} {clock(elapsed)}{eta}"


class Bar:
    """counts finished units from any thread and redraws at most every REDRAW_EVERY seconds."""

    def __init__(self, label, total, out=sys.stderr):
        self.label, self.total, self.out = label, total, out
        self.done, self.start, self.drawn = 0, time.monotonic(), 0.0
        self.lock = threading.Lock()
        self.draw()

    def draw(self, done=None):
        self.drawn = time.monotonic()
        done = self.done if done is None else done
        self.out.write(render(self.label, done, self.total, self.drawn - self.start) + CLEAR + "\r")
        self.out.flush()

    def update(self, n=1):
        with self.lock:
            self.done = min(self.done + n, self.total)
            if time.monotonic() - self.drawn >= REDRAW_EVERY:
                self.draw()

    def part(self, line):
        """advance by the share of the unit under way that a child's bar (a rendered line)
        reports, so one bar covers what several commands do."""
        m = COUNT.match(line)
        with self.lock:
            if m and int(m["total"]) and time.monotonic() - self.drawn >= REDRAW_EVERY:
                self.draw(self.done + int(m["done"]) / int(m["total"]))

    def close(self):
        """the final state as a line of its own."""
        with self.lock:
            self.out.write(render(self.label, self.done, self.total, time.monotonic() - self.start) + CLEAR + "\n")
            self.out.flush()
