# scHOTT

**scHOTT (single-cell Hierarchical Optimal Transport Tool)** is a Python package for computing **cell-to-cell distances** by integrating **gene expression data** and **gene embeddings**.  
It applies **topic modeling**, **optimal transport**, and **embedding-based distance metrics** to single-cell RNA-seq data, and provides downstream **evaluation, visualization, and clustering** tools.


## Features

The core functionality is implemented in `functions.py`:

- **Topic Modeling**
  - `fit_topics`: LDA topic modeling on expression data, outputting topic distributions and centers.  

- **Distances & Optimal Transport**
  - `sparse_ot`, `wmd`, `rwmd`, `wcd`  
  - `hott`, `hoftt`: Hierarchical Optimal Topic Transport distances  
  - `calc_costs`: Build embedding-based cost matrices  
  - `calculate_hott_dist`, `calculate_hoftt_dist`: Pairwise distance computation  

- **Evaluation**
  - `knn_evaluation`, `knn_evaluation_split`: Run classification with kNN and compute accuracy, precision, recall, and F1-score.  

- **Visualization**
  - `generate_umap_visualizations`: UMAP projection from distance matrices or raw embeddings, colored by cell type.  

- **Clustering**
  - `cluster_analysis`: K-Medoids clustering + evaluation with Silhouette Score and ARI.  
