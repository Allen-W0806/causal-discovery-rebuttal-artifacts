# Run Summary

This file summarizes the rebuttal-stage experimental runs.

## Additional Baselines

The unified baseline table includes VAR, PCMCI, VARLiNGAM, DYNOTEARS, cMLP, TCDF, GVAR, CUTS+, JRNGC, and UnCLe variants, together with the proposed method.

## Search Ablations

The search ablations compare the proposed BO-based search with:

- fixed-budget random search;
- node-wise greedy search.

These experiments are intended to verify that the improvement is not merely due to evaluating more candidate graphs, but comes from the proposed search strategy.

## Low-rank Ablation

The low-rank ablation compares the proposed method with a variant that removes the low-rank graph parameterization.

## Rank Diagnostic

The rank diagnostic reports graph-level properties of the benchmark adjacency matrices and verifies that the chosen low-rank dimension is sufficient for the tested benchmark graphs.
