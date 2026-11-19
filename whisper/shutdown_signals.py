"""Console Ctrl+C belongs to the capture/controller, not its durable-job worker."""
import signal
import threading
from contextlib import contextmanager


@contextmanager
def capture_interrupts(stop_requested):
    """Turn console interrupts into a request, so transactions cannot be torn mid-step."""
    if threading.current_thread() is not threading.main_thread():
        yield
        return
    previous = {}
    try:
        for name in ("SIGINT", "SIGBREAK"):
            value = getattr(signal, name, None)
            if value is not None:
                previous[value] = signal.signal(value, lambda *_: stop_requested.set())
        yield
    finally:
        for value, handler in previous.items():
            signal.signal(value, handler)


def ignore_worker_interrupts():
    for name in ("SIGINT", "SIGBREAK"):
        value = getattr(signal, name, None)
        if value is not None:
            signal.signal(value, signal.SIG_IGN)


@contextmanager
def shield_shutdown():
    """Ignore repeated console interrupts only while committing/closing resources."""
    if threading.current_thread() is not threading.main_thread():
        yield
        return
    previous = {}
    try:
        for name in ("SIGINT", "SIGBREAK"):
            value = getattr(signal, name, None)
            if value is not None:
                previous[value] = signal.signal(value, signal.SIG_IGN)
        yield
    finally:
        for value, handler in previous.items():
            signal.signal(value, handler)
