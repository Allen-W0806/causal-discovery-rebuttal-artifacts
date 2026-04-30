import json
import os
import numpy as np

from .eval.metrics import compute_graph_metrics


class BOReporter:
    def on_start(self, config: dict) -> None:
        pass

    def on_batch_evaluated(self, step_base: int, adjs, scores, stats_list, GT, d: int) -> None:
        pass

    def on_new_best(self, iter_idx: int, eval_count: int, best_adj: np.ndarray, scorer, GT, d: int) -> None:
        pass

    def on_end(self, result: dict, best_adj: np.ndarray | None, scorer, GT, d: int) -> dict:
        return result


class NullReporter(BOReporter):
    pass


class MinimalFinalMetricsReporter(BOReporter):
    def __init__(self, out_path: str | None = None, ignore_diag: bool = True):
        self.out_path = out_path
        self.ignore_diag = bool(ignore_diag)

    def on_end(self, result: dict, best_adj: np.ndarray | None, scorer, GT, d: int) -> dict:
        if GT is not None and best_adj is not None:
            metrics = compute_graph_metrics(GT, best_adj, ignore_diag=self.ignore_diag)
            result["best_f1"] = metrics.get("f1")
            result["best_shd"] = metrics.get("shd")
            result["best_tp"] = metrics.get("tp")
            result["best_fp"] = metrics.get("fp")
            result["best_fn"] = metrics.get("fn")
            result["best_tpr"] = metrics.get("tpr")
            result["best_fdr"] = metrics.get("fdr")
            result["best_acc"] = metrics.get("ACC")

        if self.out_path is not None:
            out_dir = os.path.dirname(self.out_path)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
            with open(self.out_path, "w") as f:
                json.dump(_to_serializable(result), f, indent=2)
        return result


def _to_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(v) for v in obj]
    return obj
