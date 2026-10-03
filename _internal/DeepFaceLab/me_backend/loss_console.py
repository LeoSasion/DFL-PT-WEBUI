"""Bounded, in-place console display for ME training loss.

The per-iteration loss history is recorded separately by the training bridge.
This class only controls the human-readable terminal line.
"""

import sys
import time


LOSS_DISPLAY_INTERVAL_SECONDS = 3.0


class LossConsole:
    def __init__(self, stream=None, clock=None, interval_seconds=LOSS_DISPLAY_INTERVAL_SECONDS):
        if interval_seconds <= 0:
            raise ValueError('Loss display interval must be positive')
        self.stream = stream if stream is not None else sys.stdout
        self.clock = clock if clock is not None else time.monotonic
        self.interval_seconds = interval_seconds
        self.last_displayed_at = None
        self.last_displayed = None
        self.latest = None
        self.live_line = False
        self.line_width = 0

    def update(self, iteration, elapsed_seconds, src_loss, dst_loss):
        self.latest = (f"[#{iteration:06d}][{elapsed_seconds * 1000:.0f}ms]"
                       f"[{src_loss:.4f}][{dst_loss:.4f}]")
        now = self.clock()
        if self.last_displayed_at is None or now - self.last_displayed_at >= self.interval_seconds:
            self._display(now)

    def _display(self, now):
        # CR returns to the same row. Pad shorter updates so no characters from
        # the prior value remain. The bridge's colorama wrapper strips ANSI
        # escapes from piped stdout, so an ANSI erase-line would not survive.
        self.stream.write('\r' + self.latest + ' ' * max(0, self.line_width - len(self.latest)))
        self.stream.flush()
        self.last_displayed_at = now
        self.last_displayed = self.latest
        self.live_line = True
        self.line_width = max(self.line_width, len(self.latest))

    def finish_line(self):
        if self.live_line:
            self.stream.write('\n')
            self.stream.flush()
            self.live_line = False
            self.line_width = 0

    def flush_latest(self):
        # A close/save can arrive between display ticks. Show the final known
        # iteration once before committing the line and printing a control log.
        if self.latest is not None and self.latest != self.last_displayed:
            self._display(self.clock())
        self.finish_line()
