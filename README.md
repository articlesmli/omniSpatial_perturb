
# A Multimodal Spatial Foundation Model & Generative Framework for Single-Cell Perturbation Modeling

## Overview

OmniSpatial-Perturb is an open-source biological AI simulator that predicts how individual cells respond to drugs or genetic edits (such as CRISPR) before running expensive physical wet-lab experiments. By combining a cell's genetics, physical shape, and surrounding tissue neighborhood, it forecasts how a treatment will shift cellular states.

Built with PyTorch and PyTorch Geometric, this end-to-end framework unifies transcriptomic representation learning, spatial morphology fusion, and conditional perturbation modeling to solve three core challenges in single-cell biology:

1. **High-Dimensional Sparseness:** Uses Masked Gene Modeling (MGM) to learn clean, reliable cell state representations across noisy or incomplete single-cell RNA-seq datasets.
2. **Spatial & Morphological Integration:** Combines cellular transcriptomics with spatial tissue coordinates and cell microscopy features using Graph Convolutional Networks (GNNs) and Cross-Attention mechanisms.
3. **Generative Perturbation Prediction:** Models cellular state shifts resulting from CRISPR knockouts or chemical treatments using a modular Latent Diffusion pipeline.

## Core Capabilities

1. **Simulate Drug & Genetic Treatments (In Silico Screening):** Feed untreated "control" cells into the Latent Diffusion engine to generate predicted post-treatment gene expression profiles for specific drugs or gene knockouts.
2. **Account for Tissue Neighborhoods:** Unlike standard models that analyse cells in isolation, OmniSpatial-Perturb builds spatial graphs to capture how neighboring cells influence treatment response—essential for studying tumors and complex organs.
3. **Fuse Multimodal Biological Data:** Aligns three distinct data streams—gene expression (RNA-seq), spatial location (coordinates), and cell appearance (vision embeddings)—into a unified representation via multi-head cross-attention.
4. **Plug-and-Play Bioinformatics Pipeline:** Natively ingests standard AnnData (.h5ad) files for streamlined fine-tuning, control-to-perturbed pairing, and automated evaluation using MSE, Pearson, and Spearman correlations.

## Practical Use Cases

1. **Drug Discovery:** Screen thousands of virtual chemical compounds on diseased tissue data to identify candidates that shift cells back toward a healthy state.
2. **Target Identification:** Simulate knocking out individual genes within a tumor microenvironment to pinpoint targets that halt cancer cell proliferation.
3. **Experimental Prioritisation:** Narrow down massive experimental search spaces to the top-performing candidates before committing time and budget at the wet-lab bench.

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
3. **Perturbation Head & Loader (`data/loader.py`, `models/perturbation.py`):** Handles AnnData normalisation, highly variable gene selection, control-to-perturbed pairing, and downstream fine-tuning/evaluation.

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
git clone https://github.com/articlesmli/omniSpatial_perturb.git
cd repository_name
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

This project is open-source and available under the terms of the [MIT License]