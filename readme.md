# NSG-MoE
Implementation for article "Modality as Heterogeneity: Node Splitting and Graph Rewiring for Multimodal Graph Learning".

## Data Preparation
Download dataset from:

Dataset provided by

- mm-graph-benchmark [1]: https://github.com/mm-graph-benchmark/mm-graph-benchmark

- MAGB [2]: https://github.com/sktsherlock/MAGB

- ABIDE [3]: https://github.com/SsGood/MMGL

<!-- - MovieLens: https://grouplens.org/datasets/movielens/ -->

[1] Mosaic of Modalities: A Comprehensive Benchmark for Multimodal Graph Learning 

[2] When Graph meets Multimodal: Benchmarking and Meditating on Multimodal Attributed Graphs Learning

[3] Multi-Modal Graph Learning for Disease Prediction

## Key parameter description
`model`: Base model, choose from `['SAGE', 'GCN', 'GAT', 'SAGE-h', 'HAN', 'HGT']`.

`mode`: Only effective when `model` in `['SAGE-h', 'HAN', 'HGT']`, choose from `['self', 'cross', 'hybrid', 'moe']`.

## Evaluation metrics
Node classification: Accuracy

Link prediction: Hits@1, Hits@3, Hits@10, MRR

## Baselines
- MMGCN: https://github.com/enoche/MMRec/blob/master/src/models/mmgcn.py
- Unigraph2: https://github.com/yf-he/UniGraph2
- MMGAT: https://github.com/tdfxlyh/MMGAT_EMO

## Run experiments

Modify the configuration in `configs/config_nc.yaml` and `configs/config_lp.yaml`.
```
cd src
python3 main_nc.py
python3 main_lp.py
```
