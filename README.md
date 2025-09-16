# scHOTT

This repository contains the code accompanying the paper:

> Fangfang Guo, Dailin Gan, Jun Li.  
> *Cell-to-cell distance that combines gene expression and gene embeddings.*  
> *Computational and Structural Biotechnology Journal*, 2024.  
> https://doi.org/10.1016/j.csbj.2024.10.044  


## Overview
This code implements **scHOTT (single-cell Hierarchical Optimal Topic Transport)**,  
a method for measuring **cell-to-cell distances** by integrating gene expression matrices with gene embeddings.  

The pipeline supports:
- **Topic modeling** with LDA on expression data  
- **Hierarchical Optimal Topic Transport (HOTT)** for distance computation  
- **kNN evaluation** (Accuracy, Precision, Recall, F1)  
- **UMAP visualization** for embedding plots  
- **K-Medoids clustering** with Silhouette and ARI metrics  

---

## Repository Contents
- `functions.py` – Core functions (topic modeling, optimal transport distance metrics, kNN evaluation, clustering, visualization).  
- `main.py` – Example pipeline to reproduce the analyses from the paper.  
