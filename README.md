# A Multimodal Spatial Foundation Model & Generative Framework for Single-Cell Perturbation Modeling

![Python](https://shields.io)
![PyTorch](https://shields.io)
![PyG](https://shields.io)
![License](https://shields.io)

---

## Overview

**OmniSpatial-Perturb** is an open-source biological AI simulator that predicts how individual cells respond to drugs or genetic edits (such as CRISPR) *before* running expensive physical wet-lab experiments. By combining a cell's genetics, physical shape, and surrounding tissue neighborhood, it forecasts how a treatment will shift cellular states.

Built with **PyTorch** and **PyTorch Geometric**, this end-to-end framework unifies transcriptomic representation learning, spatial morphology fusion, and conditional perturbation modeling to solve three core challenges in single-cell biology:

1. **High-Dimensional Sparseness:** Uses Masked Gene Modeling (MGM) to learn clean, reliable cell state representations across noisy or incomplete single-cell RNA-seq datasets.
2. **Spatial & Morphological Integration:** Combines cellular transcriptomics with spatial tissue coordinates and cell microscopy features using Graph Convolutional Networks (GCNs) and Cross-Attention mechanisms.
3. **Personalized Generative Perturbation Prediction:** Conditions a 100-step Latent Diffusion denoiser on both the untreated starting cell (`z_ctrl`) and the target perturbation (`perturb_ids`) to generate cell-specific post-treatment states.

---

## Core Capabilities

* **Simulate Drug & Genetic Treatments (*In Silico* Screening):** Feed untreated control cells into the reverse DDPM diffusion engine to generate predicted post-treatment latent states for specific drugs or CRISPR knockouts.
* **Account for Tissue Neighborhoods:** Unlike standard models that analyse cells in isolation, OmniSpatial-Perturb constructs batch k-NN spatial graphs to capture how neighbouring cells influence treatment response.
* **Fuse Multimodal Biological Data:** Aligns three distinct data streams—gene expression (RNA-seq), spatial location (`obsm['spatial']`), and cell appearance (`obsm['morphology']`)—into a unified LayerNorm-standardized representation via multi-head cross-attention.
* **Plug-and-Play Bioinformatics Pipeline:** Natively ingests standard AnnData (`.h5ad`) files for automated highly variable gene selection, individual control-to-perturbed cell pairing, and evaluation using **Latent MSE, Pearson, and Spearman correlations**.

---

## Practical Use Cases

* **Drug Discovery:** Screen thousands of virtual chemical compounds on diseased tissue data to identify candidates that shift cells back toward a healthy state.
* **Target Identification:** Simulate knocking out individual genes within a tumor microenvironment to pinpoint targets that halt cancer cell proliferation.
* **Experimental Prioritisation:** Narrow down massive experimental search spaces to the top-performing candidates before committing time and budget at the wet-lab bench.

---

## Architecture Overview

    [ Single-Cell RNA-seq ] ──► Masked Gene Transformer ──┐
                                                          ├─► Cross-Attention ──► LayerNorm ──► Conditional Latent ──► Predicted Perturbed
    [ Spatial Coordinates ] ──► Spatial GNN (PyG k-NN)  ──┤   Fusion Head         (z_ctrl)      Diffusion Denoiser     Cell State
                                                          │
    [ Cell Morphology ]    ──► Vision Features  ──────────┘

The pipeline operates in modular stages:

1. **Foundation Backbone (`models/transformer.py`):** Encodes binned gene expression tokens and applies masked language modeling objectives to extract 256-dimensional transcriptomic representations.
2. **Spatial Graph Integration (`models/spatial_gnn.py`):** Constructs local k-NN spatial neighborhood graphs and aligns transcriptomic embeddings with morphological features via multi-head cross-attention (`SpatialCrossAttentionFusion`).
3. **Conditional Perturbation Head (`models/perturbation.py`):** Concatenates noisy target states (`z_t`), multimodal untreated control embeddings (`z_ctrl`), perturbation embeddings (`perturb_ids`), and diffusion timesteps (`t`) to predict reverse denoising steps.
4. **Multimodal Loader & Pipeline (`data/loader.py`, `scripts/`):** Handles AnnData normalisation, integer perturbation mapping, single-cell control pairing, and joint fine-tuning/evaluation.

---

## Data Sources & Preparation

OmniSpatial-Perturb supports standard AnnData (`.h5ad`) single-cell and spatial perturbation datasets:

* **Foundation Pretraining (`pbmc3k.h5ad`):** Uses the **10x Genomics PBMC 3k** dataset (2,700 healthy human Peripheral Blood Mononuclear Cells across 32,738 genes) for Masked Gene Modeling pretraining.
* **Drug & Cytokine Perturbation Fine-Tuning (`perturbation_data.h5ad`):** Uses **Kang et al. (2018)** (*Nature Biotechnology*) human PBMC single-cell data comparing untreated control cells (`ctrl`) against interferon-beta (`IFN_beta`) stimulated cells.
* **Multimodal Spatial & Morphology Fallbacks:** When spatial coordinates (`adata.obsm['spatial']`) or cell microscopy embeddings (`adata.obsm['morphology']`) are present in spatial transcriptomics datasets (e.g., 10x Visium or Xenium), `PerturbationDataset` ingests them directly. For suspension Perturb-seq datasets without imaging layers, `data/loader.py` supplies zero-tensor fallbacks so the full multimodal architecture runs seamlessly.

### Download Real Perturbation Data (`Kang et al., 2018`)

Install `pertpy` and generate `perturbation_data.h5ad`:

    pip install pertpy
    python -c "import pertpy as pt, scanpy as sc; adata = pt.data.kang_2018(); adata.obs['perturbation'] = adata.obs['label'].map({'ctrl': 'control', 'stim': 'IFN_beta'}).astype(str); sc.pp.subsample(adata, n_obs=5000, random_state=42); adata.write_h5ad('perturbation_data.h5ad')"

---

## Project Structure

    omniSpatial_perturb/
    ├── .github/
    │   └── workflows/
    │       └── ci_cd.yaml
    ├── data/
    │   ├── __init__.py
    │   └── loader.py                           # Multimodal AnnData & perturbation pair loader
    ├── models/
    │   ├── __init__.py
    │   ├── transformer.py                      # Foundation model (Masked Gene Modeling)
    │   ├── spatial_gnn.py                      # Spatial k-NN GCN & cross-attention fusion
    │   └── perturbation.py                     # Control-conditioned Latent Diffusion head
    ├── scripts/
    │   ├── fine_tune.py                        # Joint multimodal diffusion fine-tuning loop
    │   ├── evaluate.py                         # Reverse DDPM sampling & correlation metrics
    │   └── train_pretrain.py                   # Pretraining script stub
    ├── tests/
    │   ├── test_models.py                      # Core architecture unit tests
    │   └── test_diffusion.py                   # Diffusion conditioning, gradient & overfit tests
    ├── Dockerfile
    ├── LICENSE                                 # MIT License
    ├── README.md                               # Project documentation
    ├── pyproject.toml
    ├── requirements.txt                        # Dependency list
    ├── setup.py
    ├── omnispatial_transformer_pretrained.pth
    ├── omnispatial_multimodal_finetuned.pth
    ├── pbmc3k.h5ad
    └── perturbation_data.h5ad

---

## Quick Start

### Installation

Clone the repository and install dependencies:

    git clone https://github.com/articlesmli/omniSpatial_perturb.git
    cd omniSpatial_perturb
    pip install -r requirements.txt

### Run Unit Tests

Verify model shapes, gradient flow, single-batch overfitting, and control-state personalization:

    PYTHONPATH=. python tests/test_diffusion.py

### Fine-Tuning & Evaluation

Run the joint multimodal fine-tuning pipeline on your AnnData (`.h5ad`) perturbation dataset:

    PYTHONPATH=. python -u scripts/fine_tune.py

Run 100-step reverse DDPM sampling to evaluate validation Latent MSE and cell-wise Pearson/Spearman correlation coefficients:

    PYTHONPATH=. python -u scripts/evaluate.py

---

## Technical Stack

* **Core ML:** PyTorch, PyTorch Geometric
* **Bioinformatics:** Scanpy, AnnData, Pertpy, SciPy
* **Generative Modeling & Evaluation:** Control-Conditioned Latent Diffusion (DDPM), MSE, Pearson & Spearman correlation metrics

---

## License

This project is open-source and available under the terms of the [MIT License](LICENSE).
