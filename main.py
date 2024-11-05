adata = sc.read_h5ad("sample_aorta_data_updated.h5ad")

GPT_DIM=3072
hvg_names = pd.read_csv('Aorta_3000hvg_0711.csv', usecols=['hvg'])
hvg_list=list(hvg_names.iloc[:,0])
gene_list=list(adata.var.index)

gene_embeddings_full_matrix=np.load('Aorta_allgenes_embedding_0527.npy')
gene_embeddings_matrix=np.load('Aorta_3000hvg_embedding_0711.npy')


indices = [gene_list.index(item) for item in hvg_list if item in gene_list]

expr_matrix=adata.X[:,indices]
expr_full_matrix=adata.X


celltype = adata.obs.celltype

known_id=np.where(adata.obs.celltype!='Unknown')[0]
expr_matrix_known = expr_matrix[known_id,:]
celltype_known = celltype.iloc[known_id]  

for SEED in range(1, 15):
    print(f"Running with SEED = {SEED}")
    data=calc_costs(expr_matrix, gene_embeddings_matrix, hvg_list=NULL, p=1, K_lda=90, n_words_keep = 50,seed=SEED)
    
    C = data['cost_T']
    topic_proportions = data['proportions']
    
    # 初始化 hott_dist 矩阵
    hott_dist = np.zeros((topic_proportions.shape[0], topic_proportions.shape[0]))
    # 并行计算 hott_dist 矩阵上三角部分
    results = Parallel(n_jobs=-1)(delayed(calculate_hott_dist)(i, j) for i in range(topic_proportions.shape[0]) for j in range(i + 1, topic_proportions.shape[0]))
    # 将结果填入 hott_dist 矩阵
    index = 0
    for i in range(topic_proportions.shape[0]):
        for j in range(i + 1, topic_proportions.shape[0]):
            hott_dist[i, j] = results[index]
            index += 1
    # 对称填充矩阵
    hott_dist = hott_dist + hott_dist.T
    print("并行计算完成。")
    hott_dist_known=hott_dist[np.ix_(known_id, known_id)]
    
    MC_full=expr_full_matrix@gene_embeddings_full_matrix/len(gene_list)
    MC_full_known=MC_full[known_id,:]
    
    pca = PCA(n_components=50)
    pca_result = pca.fit_transform(expr_matrix)
    
    knn_evaluation(X=None, y=celltype_known, distance_matrix=hott_dist_known, test_size=0.2, k=10,seed=SEED)
    knn_evaluation(X=MC_full_known, y=celltype_known, distance_matrix=None, test_size=0.2, k=10,seed=SEED)
    knn_evaluation(X=expr_matrix_known, y=celltype_known, distance_matrix=None, test_size=0.2, k=10,seed=SEED)
    
    generate_umap_visualizations(celltype, dist_matrix=hott_dist, data=None, n_neighbors=15, min_dist=0.1, embedding_suffix =f"0917_HOTT_Aorta_UMAP_embedding_{SEED}", fig_suffix=f"0917_HOTT_Aorta_UMAP_{SEED}")
    generate_umap_visualizations(celltype, dist_matrix=None, data=MC_full, n_neighbors=15, min_dist=0.1, embedding_suffix =f"0917_MCD_Aorta_UMAP_embedding_{SEED}", fig_suffix=f"0917_MCD_Aorta_UMAP_{SEED}")
    generate_umap_visualizations(celltype, dist_matrix=None, data=pca_result, n_neighbors=15, min_dist=0.1, embedding_suffix =f"0917_Aorta_UMAP_embedding_{SEED}", fig_suffix=f"0917_Aorta_UMAP_{SEED}")
    
    cluster_analysis(celltype, hott_dist, squareform(pdist(MC_full, metric='cosine')), squareform(pdist(pca_result, metric='cosine')), seed=SEED)
    print(f"Completed SEED = {SEED}\n")

print("All SEED runs completed.")
