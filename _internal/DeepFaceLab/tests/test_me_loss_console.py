from io import StringIO

from me_backend.loss_console import LossConsole


def test_loss_console_throttles_and_overwrites_one_row():
    stream = StringIO()
    now = [0.0]
    console = LossConsole(stream=stream, clock=lambda: now[0], interval_seconds=3.0)
    console.update(1, 0.010, 0.5, 0.6)
    now[0] = 1.0
    console.update(2, 0.011, 0.4, 0.5)
    now[0] = 2.9
    console.update(3, 0.012, 0.3, 0.4)
    assert stream.getvalue() == '\r[#000001][10ms][0.5000][0.6000]'
    now[0] = 3.0
    console.update(4, 0.013, 0.2, 0.3)
    assert stream.getvalue().endswith('\r[#000004][13ms][0.2000][0.3000]')
    assert stream.getvalue().count('\n') == 0
    assert stream.getvalue().count('\r') == 2


def test_loss_console_flushes_latest_before_save_and_does_not_duplicate_line():
    stream = StringIO()
    now = [0.0]
    console = LossConsole(stream=stream, clock=lambda: now[0], interval_seconds=3.0)
    console.update(1, 0.1, 0.5, 0.6)
    now[0] = 1.0
    console.update(2, 0.1, 0.4, 0.5)
    console.flush_latest()
    assert stream.getvalue().endswith('\r[#000002][100ms][0.4000][0.5000]\n')
    before = stream.getvalue()
    console.flush_latest()
    assert stream.getvalue() == before
    now[0] = 1.1
    console.update(3, 0.1, 0.3, 0.4)
    assert stream.getvalue() == before  # The interval is still measured from the last display.
    now[0] = 4.0
    console.update(4, 0.1, 0.2, 0.3)
    assert stream.getvalue().endswith('\r[#000004][100ms][0.2000][0.3000]')


def test_loss_console_clears_tail_when_new_metric_is_shorter():
    stream = StringIO()
    now = [0.0]
    console = LossConsole(stream=stream, clock=lambda: now[0], interval_seconds=3.0)
    console.update(1, 1.0, 0.5, 0.6)
    old = '[#000001][1000ms][0.5000][0.6000]'
    now[0] = 3.0
    console.update(2, 0.01, 0.4, 0.5)
    new = '[#000002][10ms][0.4000][0.5000]'
    assert stream.getvalue() == '\r' + old + '\r' + new + ' ' * (len(old) - len(new))


def test_loss_console_rejects_invalid_interval():
    try:
        LossConsole(interval_seconds=0)
    except ValueError:
        pass
    else:
        raise AssertionError('Zero-second display interval must be rejected')
