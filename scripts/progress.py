"""a one-line progress bar on stderr, redrawn in place.

each redraw ends in a carriage return and its last line in a newline, so a
parent reading the output (hierophant_pipeline's Run.sh) can tell the two
apart: it draws redraws in place and keeps only finished lines in its logs.

an outer bar (a round, a whole sweep) is not drawn by itself: while it is open
it prefixes every bar drawn, briefly, and it ends on a line of its own.
"""

import re
import shutil
import sys
import threading
import time

WIDTH = 24
REDRAW_EVERY = 0.2  # seconds between redraws
CLEAR = "\x1b[K"  # erase to the end of the line, past a longer earlier redraw
SEP = " | "  # after each outer bar's brief count
OPEN = []  # the outer bars open, outermost first


def clock(seconds):
    m, s = divmod(int(seconds), 60)
    return f"{m // 60}:{m % 60:02d}:{s:02d}" if m >= 60 else f"{m}:{s:02d}"


# a rendered bar's own count, after the outer bars' briefs
COUNT = re.compile(r"(?:.*\| )?(?P<label>[^|]*?) \[[#-]*\] (?P<done>\d+)/(?P<total>\d+) ")


def around():
    """the open outer bars, briefly: 'round 2 select 1/4 3:10 | '."""
    return "".join(b.brief() + SEP for b in OPEN)


def fit(line):
    """cut to the terminal's width: a redraw that wraps cannot be drawn over. a pipe is
    not cut, so a parent sees the whole line."""
    return line[:shutil.get_terminal_size((sys.maxsize, 0)).columns - 1]


def render(label, done, total, elapsed):
    """'label [#####-------] 12/40 30% 0:14 eta 0:33'. done may be fractional, a unit
    under way; the count shows only finished ones."""
    frac = min(done / total, 1.0) if total else 1.0
    fill = round(WIDTH * frac)
    eta = f" eta {clock(elapsed / done * (total - done))}" if 0 < done < total else ""
    return f"{label} [{'#' * fill}{'-' * (WIDTH - fill)}] {int(done)}/{total} {frac:.0%} {clock(elapsed)}{eta}"


class Bar:
    """counts finished units from any thread and redraws at most every REDRAW_EVERY seconds.
    an outer one is opened with `with`, and named units of it with stage()."""

    def __init__(self, label, total, out=sys.stderr, outer=False):
        self.label, self.total, self.out, self.outer = label, total, out, outer
        self.done, self.start, self.drawn, self.note = 0, time.monotonic(), 0.0, None
        self.lock = threading.Lock()
        if not outer:
            self.draw()

    def name(self):
        return f"{self.label} {self.note}" if self.note else self.label

    def brief(self):
        return f"{self.name()} {int(self.done)}/{self.total} {clock(time.monotonic() - self.start)}"

    def write(self, done, end):
        line = around() + render(self.name(), done, self.total, time.monotonic() - self.start)
        self.out.write(fit(line) + CLEAR + end)
        self.out.flush()

    def draw(self, done=None):
        self.drawn = time.monotonic()
        self.write(self.done if done is None else done, "\r")

    def update(self, n=1):
        with self.lock:
            self.done = min(self.done + n, self.total)
            if not self.outer and time.monotonic() - self.drawn >= REDRAW_EVERY:
                self.draw()

    def stage(self, note):
        """the next unit, named: the one before it is finished."""
        with self.lock:
            if self.note:
                self.done = min(self.done + 1, self.total)
            self.note = note

    def __enter__(self):
        OPEN.append(self)
        return self

    def __exit__(self, kind, *_):
        OPEN.remove(self)
        if kind is None:
            self.done, self.note = self.total, None
        self.close()

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
            self.write(self.done, "\n")


class Quiet:
    """an outer bar that draws nothing, for code that counts either way."""

    def update(self, n=1):
        pass

    def stage(self, note):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass


def outer(on, label, total, out=sys.stderr):
    return Bar(label, total, out, outer=True) if on else Quiet()
