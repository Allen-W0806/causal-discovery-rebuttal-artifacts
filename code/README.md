# Anonymized NC8 Rebuttal Code Snapshot

## Environment used for our run

CPU: AMD Ryzen Threadripper PRO 7965WX 24-Cores  
GPU: NVIDIA RTX A6000  
Driver: 560.35.03  
PyTorch: 2.1.0+cu126  
CUDA: 12.6  

## Install

```bash
pip install -r requirements.txt
```

## Run

From the repository root:

```bash
# Low-rank BO
bash scripts/run_nc8.sh 0 0

# Node-wise greedy
bash scripts/run_nodewise_greedy_nc8.sh 0 0

# Node-wise exhaustive
bash scripts/run_nodewise_exhaustive_nc8.sh 0 0
```

The first argument is replica id and the second argument is seed. For the reported run, use seed 0 and replicas 0--4.

Optional:

```bash
bash scripts/run_all_replicas_nc8.sh 0
```

## Fixed settings

- dataset: NC8
- lag: 16
- eval: 8000
- batch size: 32
- seed: 0
- replicas: 0--4
