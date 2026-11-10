# **A Multimodal Spatial Foundation Model & Generative Framework for Single-Cell Perturbation Modeling**

---

## Overview

**OmniSpatial-Perturb** is an end-to-end biological AI framework designed to unify transcriptomic representation learning, spatial morphology fusion, and conditional perturbation modeling. Built using PyTorch and PyTorch Geometric, it addresses key challenges in single-cell biology:

- **High-Dimensional Sparseness:** Uses Masked Gene Modeling (MGM) to learn robust cell state representations across heterogeneous single-cell RNA-seq datasets.
- **Spatial & Morphological Integration:** Combines cellular transcriptomics with spatial coordinates and cell image representations using Graph Convolutional Networks (GNNs) and Cross-Attention mechanisms.
- **Generative Perturbation Prediction:** Models cellular state shifts resulting from CRISPR knockouts or drug treatments using modular fine-tuning and latent evaluation pipelines.

---

## Architecture Overview

```text
[ Single-Cell RNA-seq ] ──► Masked Gene Transformer ──┐
                                                      ├─► Cross-Attention ──► Latent Diffusion ──► Predicted Perturbed
[ Spatial Coordinates ] ──► Spatial GNN (PyG)  ───────┤   Fusion Head         Engine               Cell State
                                                      │
[ Cell Morphology ]    ──► Vision Features  ──────────┘

```

The pipeline operates in modular stages:

1. **Foundation Backbone (`models/transformer.py`):** Encodes binned gene expression tokens and applies masked language modeling objectives to handle out-of-distribution biological variance.
2. **Spatial Graph Integration (`models/spatial_gnn.py`):** Constructs local spatial neighborhood graphs ($k$-NN / Delaunay) and aligns gene expression embeddings with morphological features via multi-head cross-attention.
3. **Perturbation Head & Loader (`data/loader.py`, `models/perturbation.py`):** Handles AnnData normalization, highly variable gene selection, control-to-perturbed pairing, and downstream fine-tuning/evaluation.

---

## Project Structure

```text
omnispatial_perturb/
├── .github/
│   └── workflows/
│       └── ci_cd.yaml
├── data/
│   ├── __init__.py
│   └── loader.py                 # AnnData & Perturbation dataset loader
├── models/
│   ├── __init__.py
│   ├── transformer.py          # Foundation model (Masked Gene Modeling)
│   ├── spatial_gnn.py          # Spatial omics & cross-attention integration
│   └── perturbation.py         # Diffusion / VAE state transition head
├── scripts/
│   ├── fine_tune.py            # Perturbation fine-tuning loop
│   ├── evaluate.py             # Validation metric evaluation (MSE, Pearson, Spearman)
│   └── train_pretrain.py       # Pretraining script stub
├── tests/
│   └── test_models.py
├── LICENSE                     # MIT License
├── README.md                   # Project documentation
├── requirements.txt            # Dependency list
├── setup.py
├── pyproject.toml
├── omnispatial_transformer_pretrained.pth
├── omnispatial_transformer_finetuned.pth
├── pbmc3k.h5ad
└── perturbation_data.h5ad

```

---

## Quick Start

### Installation

Clone the repository and install dependencies:

```bash
git clone [https://github.com/articlesmli/omniSpatial_perturb.git](https://github.com/articlesmli/omniSpatial_perturb.git)
cd omniSpatial_perturb
pip install -r requirements.txt

```

### Fine-Tuning & Evaluation

Run the fine-tuning pipeline on your AnnData (`.h5ad`) perturbation dataset:

```bash
PYTHONPATH=. python -u scripts/fine_tune.py

```

Evaluate validation loss and gene-wise Pearson/Spearman correlation coefficients:

```bash
PYTHONPATH=. python -u scripts/evaluate.py

```

---

## Technical Stack

* **Core ML:** PyTorch, PyTorch Geometric
* **Bioinformatics:** Scanpy, AnnData, SciPy
* **Generative Modeling & Evaluation:** Score-based Latent Diffusion, Pearson/Spearman correlation metrics

---

## License

This project is open-source and available under the terms of the [MIT License](https://www.google.com/search?q=LICENSE).

```