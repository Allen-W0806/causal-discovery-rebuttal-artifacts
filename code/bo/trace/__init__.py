"""Trace helpers for BO instrumentation."""

from .meta_trace import (
    append_jsonl,
    collect_grad_stats,
    collect_param_stats,
    collect_weight_stats,
    dump_top_edges,
    flatten_offdiag,
    param_delta_from_snapshot,
    snapshot_params,
)

__all__ = [
    "append_jsonl",
    "collect_grad_stats",
    "collect_param_stats",
    "collect_weight_stats",
    "dump_top_edges",
    "flatten_offdiag",
    "param_delta_from_snapshot",
    "snapshot_params",
]
