"""progress bars on a terminal, redrawn in place.

each redraw ends in a carriage return and a bar's last line in a newline, so a
parent reading the output (hierophant_pipeline's Run.sh) can tell the two
apart: it draws redraws in place and keeps only finished lines in its logs.

one bar is drawn at a time. an outer bar (an eval's whole sweep, a round of the
automated run) is opened with `with`; a bar opened inside it is not drawn but
moves it, and only its finished line is written, above. so the drawn share
never jumps back when the work inside it starts over.
"""

import re
import shutil
import sys
import threading
import time

WIDTH = 24
REDRAW_EVERY = 0.2  # seconds between redraws
CLEAR = "\x1b[K"  # erase to the end of the line, past a longer earlier redraw
OUTER_DIGITS = 1  # an outer bar's share in tenths of a percent: its units are long
OPEN = []  # the outer bars open, outermost first
SCREEN = threading.RLock()  # one write to the terminal at a time
shown = None  # the bar drawn in place, which a line written above it redraws below

# a rendered bar: its label, count, share and, after " | ", what it is doing
COUNT = re.compile(r"(?P<label>[^\[]*?) \[[#-]*\] (?P<done>\d+)/(?P<total>\d+) (?P<pct>[\d.]+)%"
                   r"(?:[^|]*\| (?P<detail>.*))?")


def clock(seconds):
    m, s = divmod(int(seconds), 60)
    return f"{m // 60}:{m % 60:02d}:{s:02d}" if m >= 60 else f"{m}:{s:02d}"


def fit(line):
    """cut to the terminal's width: a redraw that wraps cannot be drawn over. a pipe is
    not cut, so a parent sees the whole line."""
    return line[:shutil.get_terminal_size((sys.maxsize, 0)).columns - 1]


def render(label, done, total, elapsed, digits=0, detail=""):
    """'label [#####-------] 12/40 30% 0:14 eta 0:33 | detail'. done may be fractional, a
    unit under way; the count shows only finished ones."""
    frac = min(done / total, 1.0) if total else 1.0
    fill = round(WIDTH * frac)
    eta = f" eta {clock(elapsed / done * (total - done))}" if 0 < done < total else ""
    return (f"{label} [{'#' * fill}{'-' * (WIDTH - fill)}] {int(done)}/{total} {frac:.{digits}%} "
            f"{clock(elapsed)}{eta}" + (f" | {detail}" if detail else ""))


class Bar:
    """counts finished units from any thread and redraws at most every REDRAW_EVERY seconds.
    an outer one is opened with `with`; its units are the bars opened inside it, or the
    stages it is told of."""

    def __init__(self, label, total, out=sys.stderr, outer=False, detail=""):
        self.label, self.total, self.outer = label, total, outer
        self.out = getattr(out, "raw", out)
        self.parent = None if outer or not OPEN else OPEN[-1]
        self.done, self.start, self.drawn, self.note = 0, time.monotonic(), 0.0, None
        self.frac, self.detail, self.staged = 0.0, detail, 0
        self.lock = threading.RLock()
        self.draw()

    def name(self):
        return f"{self.label} {self.note}" if self.note else self.label

    def line(self, done):
        return fit(render(self.name(), done, self.total, time.monotonic() - self.start,
                          OUTER_DIGITS if self.outer else 0, self.detail))

    def paint(self):
        global shown
        with SCREEN:
            self.out.write(self.line(self.done + self.frac) + CLEAR + "\r")
            self.out.flush()
            shown = self

    def draw(self):
        """in place, or inside an outer bar as the share of its unit under way."""
        self.drawn = time.monotonic()
        if self.parent:
            self.parent.follow((self.done + self.frac) / self.total if self.total else 1.0, self.detail or self.label)
        else:
            self.paint()

    def due(self):
        return time.monotonic() - self.drawn >= REDRAW_EVERY

    def update(self, n=1):
        with self.lock:
            self.done, self.frac = min(self.done + n, self.total), 0.0
            if self.due():
                self.draw()

    def follow(self, frac, detail=None):
        """the unit under way is frac finished (never less than it was), doing detail. a
        new detail is drawn at once: the work it names may outlast the next redraw."""
        with self.lock:
            self.frac = max(self.frac, min(frac, 1.0))
            changed = detail is not None and detail != self.detail
            if changed:
                self.detail = detail
            if changed or self.due():
                self.draw()

    def stage(self, note):
        """the next unit, named: the ones before it are finished."""
        with self.lock:
            self.done, self.frac, self.detail = min(self.staged, self.total), 0.0, ""
            self.staged, self.note = self.staged + 1, note
            self.draw()

    def __enter__(self):
        OPEN.append(self)
        return self

    def __exit__(self, kind, *_):
        OPEN.remove(self)
        if kind is None:
            self.done, self.note = self.total, None
        self.close()

    def close(self):
        """the final state as a line of its own, above the bar drawn in place if another is."""
        global shown
        with self.lock:
            self.frac, self.detail = 0.0, ""
            if self.parent:
                self.parent.update()
            with SCREEN:
                live = shown is not None and shown is not self and shown.out.isatty()
                if live:
                    shown.out.write("\r" + CLEAR)
                self.out.write(self.line(self.done) + CLEAR + "\n")
                self.out.flush()
                if shown is self:
                    shown = None
                elif live:
                    shown.paint()


class Above:
    """a stream whose lines are written above the bar drawn in place on the terminal,
    which is redrawn below them."""

    def __init__(self, raw):
        self.raw, self.mid = raw, False

    def write(self, s):
        with SCREEN:
            bar = shown
            live = bar is not None and bar.out.isatty()
            if live and not self.mid:
                bar.out.write("\r" + CLEAR)
            n = self.raw.write(s)
            if live:
                self.mid = not s.endswith("\n")
                if not self.mid:
                    self.raw.flush()
                    bar.paint()
            return n

    def flush(self):
        self.raw.flush()

    def __getattr__(self, name):
        return getattr(self.raw, name)


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
