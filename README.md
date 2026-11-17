# DVD-Rental-customer-segmentation
Customer segmentation using PostgreSQL, K-Means clustering and Principal Component Analysis (PCA).

Customer Segmentation with KMeans
=================================

What this folder contains
-------------------------
This portfolio package documents a customer clustering analysis on a
per-customer rental dataset (ratings, spend, and genre mix).

  README.txt          - this file
  report.tex          - LaTeX source for the PDF findings report (Overleaf)
  figures/            - PNG plots embedded by report.tex

  Upstream analysis script (one folder up from portfolio/):
  DatasetPerCustomerAnalysis.py
  DatasetPerCustomerTable.csv

What the analysis does
----------------------
1. Load one row per customer (599 customers, 18 features).
2. Standardize features with StandardScaler.
3. Evaluate KMeans for k = 2..19; pick k with the highest silhouette score
   (here: k = 14, silhouette ~ 0.059).
4. Cross-check with Gaussian Mixture and Agglomerative (Ward) clustering.
5. Use GMM BIC/AIC (here BIC prefers k = 2) and a Ward dendrogram.
6. Fit final KMeans; interpret segments with PCA, a colour legend table,
   a standardized profile heatmap, and silhouette diagrams (also k = 5
   for contrast).

Plots in figures/
-----------------
  elbow_silhouette.png      Elbow (inertia) + silhouette vs k
  algorithm_comparison.png  Silhouette by algorithm
  gmm_bic_aic.png           GMM BIC and AIC
  dendrogram.png            Ward hierarchical dendrogram
  pca_clusters.png          Clusters in 2D PCA space
  cluster_legend.png        Plain-language cluster descriptions
  cluster_heatmap.png       Cluster profiles (std. deviations)
  silhouette_best_k.png     Per-sample silhouette for chosen k
  silhouette_k5.png         Per-sample silhouette for k = 5

How to regenerate the plots
---------------------------
From the project root (where DatasetPerCustomerAnalysis.py lives):

  set MPLBACKEND=Agg
  python DatasetPerCustomerAnalysis.py

On PowerShell:

  $env:MPLBACKEND='Agg'
  python DatasetPerCustomerAnalysis.py

Figures are written to this folder under figures/. Set SAVE_FIGURES=0 to
skip saving PNGs.

How to build the PDF report (Overleaf)
--------------------------------------
1. Create a new Overleaf project.
2. Upload report.tex and the figures/ folder (keep the same names).
3. Compile. Download the PDF for your portfolio / applications.

Requirements
------------
Python 3.x with: pandas, numpy, matplotlib, scipy, scikit-learn

Key result snapshot (from last run)
-----------------------------------
  Customers: 599
  Chosen k (KMeans silhouette): 14
  Best silhouette: ~0.059 (soft / overlapping segments)
  GMM lowest BIC: k = 2
  Stand-out segment: Cluster 9 — high ratings (~4.74), high spend (~$1959),
  ~11% of customers; sports / animation / scifi lean.
