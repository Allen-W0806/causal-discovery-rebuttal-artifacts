import csv
import os


class RunLogger:
    def __init__(self, run_dir: str):
        self.run_dir = run_dir
        self.figures_dir = os.path.join(run_dir, "figures")
        os.makedirs(self.run_dir, exist_ok=True)
        os.makedirs(self.figures_dir, exist_ok=True)

        self.progress_path = os.path.join(run_dir, "progress.csv")
        self.history_path = os.path.join(run_dir, "history.csv")

        self._progress_required = [
            "true_evals_so_far",
            "total_score",
            "fit_term",
            "mse",
        ]
        self._history_required = [
            "true_evals_so_far",
            "best_total_score",
            "best_fit_term",
            "best_mse",
        ]

    def log_progress(self, row: dict):
        self._write_row(self.progress_path, row, self._progress_required)

    def log_history(self, row: dict):
        self._write_row(self.history_path, row, self._history_required)

    def _write_row(self, path: str, row: dict, required_cols: list[str]):
        row = dict(row or {})
        for key in required_cols:
            row.setdefault(key, "")

        if not os.path.exists(path):
            fieldnames = required_cols + sorted([k for k in row.keys() if k not in required_cols])
            self._write_new(path, fieldnames, [row])
            return

        existing = self._read_header(path)
        if not existing:
            fieldnames = required_cols + sorted([k for k in row.keys() if k not in required_cols])
            self._write_new(path, fieldnames, [row])
            return

        new_keys = [k for k in row.keys() if k not in existing]
        if new_keys:
            fieldnames = existing + sorted(new_keys)
            rows = self._read_rows(path)
            self._write_new(path, fieldnames, rows)
        else:
            fieldnames = existing

        with open(path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writerow(row)

    @staticmethod
    def _read_header(path: str):
        try:
            with open(path, "r", newline="") as f:
                reader = csv.reader(f)
                return next(reader, None)
        except FileNotFoundError:
            return None

    @staticmethod
    def _read_rows(path: str):
        try:
            with open(path, "r", newline="") as f:
                reader = csv.DictReader(f)
                return list(reader)
        except FileNotFoundError:
            return []

    @staticmethod
    def _write_new(path: str, fieldnames: list[str], rows: list[dict]):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
