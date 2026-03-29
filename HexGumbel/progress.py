"""Small tqdm wrapper with a no-op fallback."""

from __future__ import annotations

import sys


class NullProgress:
    def update(self, n: int = 1):
        return None

    def set_postfix(self, *args, **kwargs):
        return None

    def set_total(self, total):
        return None

    def close(self):
        return None


def _has_tty() -> bool:
    return bool(getattr(sys.stdout, "isatty", lambda: False)() or getattr(sys.stderr, "isatty", lambda: False)())


def make_progress(total, desc: str, unit: str, enabled: bool, leave: bool = True):
    if not enabled or not _has_tty():
        return NullProgress()

    try:
        from tqdm.auto import tqdm
    except Exception:
        return NullProgress()

    class TqdmProgress:
        def __init__(self):
            self._bar = tqdm(
                total=total,
                desc=desc,
                unit=unit,
                leave=leave,
                file=sys.stdout,
                dynamic_ncols=True,
                smoothing=0.1,
                mininterval=0.2,
            )

        def update(self, n: int = 1):
            self._bar.update(n)

        def set_postfix(self, *args, **kwargs):
            self._bar.set_postfix(*args, **kwargs)

        def set_total(self, total):
            if total is None:
                return
            target = max(int(total), int(self._bar.n))
            if self._bar.total != target:
                self._bar.total = target
                self._bar.refresh()

        def close(self):
            self._bar.close()

    return TqdmProgress()
