# monitor.py
import time
from collections import defaultdict
from contextlib import contextmanager
import json
from pathlib import Path

class Monitor:
    def __init__(self):
        self.timings = defaultdict(list)
        self._start_times = {}

    def start(self, section: str):
        self._start_times[section] = time.time()

    def stop(self, section: str):
        if section not in self._start_times:
            raise ValueError(f"Section '{section}' was never started.")
        duration = time.time() - self._start_times.pop(section)
        self.timings[section].append(duration)
        return duration

    @contextmanager
    def section(self, name: str):
        self.start(name)
        try:
            yield
        finally:
            self.stop(name)

    def get_stats(self):
        """Return raw stats as a dict."""
        return {
            sec: {
                "count": len(times),
                "total": sum(times),
                "avg": sum(times) / len(times),
            }
            for sec, times in self.timings.items()
        }

    def report(self, file_path: str | Path = None):
        """Print or save summary report."""
        stats = self.get_stats()

        if file_path is None:
            # Print to console
            print("\n===== Timing Report =====")
            for section, s in stats.items():
                print(f"{section:15s} | calls: {s['count']:3d} | "
                      f"avg: {s['avg']:.4f}s | total: {s['total']:.4f}s")
        else:
            file_path = Path(file_path)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(file_path, "w") as f:
                json.dump(stats, f, indent=2)
