import numpy as np
import pandas as pd
from gensim import corpora
from gensim.models import LdaModel
import csv
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from joblib import Parallel, delayed
import ot
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
import umap
from sklearn.decomposition import PCA
import scanpy as sc
import matplotlib.colors as mcolors
from sklearn.preprocessing import StandardScaler
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
from sklearn_extra.cluster import KMedoids
from sklearn.metrics import silhouette_score, adjusted_rand_score
import os

def fit_topics(data, embeddings, vocab, K, seed):
    """Fit a topic model to gene expression data using gensim."""
    
    # Convert data to document-word frequency format
    corpus = []
    for expression_values in data:
        doc = [(i, freq) for i, freq in enumerate(expression_values) if freq > 0]
        corpus.append(doc)
    
    # Create dictionary
    dictionary = corpora.Dictionary([[gene] for gene in vocab])
    
    # Train LDA model
    model = LdaModel(corpus, num_topics=K, id2word=dictionary, passes=100, iterations=2000, random_state=seed)
    
    # Extract topic-word matrix
    topics = np.zeros((K, len(vocab)))
    for i in range(K):
        for word_id, prob in model.get_topic_terms(i, topn=len(vocab)):
            topics[i, word_id] = prob
    
    # Compute topic centers
    lda_centers = np.matmul(topics, embeddings)
    
    topics1 = model.show_topics(num_topics=K, num_words=5, log=False, formatted=False)

    with open(f'0917_Aorta_topics_seed_{seed}.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Topic', 'Word', 'Weight'])
        for topic in topics1:
            topic_num = topic[0]
            for word, weight in topic[1]:
                writer.writerow([topic_num, word, weight])

    # Extract document-topic distributions
    topic_proportions = np.zeros((len(corpus), K))
    
    for i, bow in enumerate(corpus):
        doc_topics = model.get_document_topics(bow, minimum_probability=0)
        for topic_id, prob in doc_topics:
            topic_proportions[i, topic_id] = prob

    np.savetxt(f'0917_Aorta_topic_proportions_seed_{seed}.csv', topic_proportions, delimiter=',', fmt='%f')

    print("Matrix topic_proportions has been saved to topic_proportions_full25.csv file.")

    return topics, lda_centers, topic_proportions


def sparse_ot(weights1, weights2, M):
    """ Compute transport for (possibly) un-normalized sparse distributions"""
    
    weights1 = weights1 / weights1.sum()
    weights2 = weights2 / weights2.sum()
    
    active1 = np.where(weights1)[0]
    active2 = np.where(weights2)[0]
    
    weights_1_active = weights1[active1]
    weights_2_active = weights2[active2]
    M_reduced = np.ascontiguousarray(M[active1][:, active2])
    
    return ot.emd2(weights_1_active, weights_2_active, M_reduced)


def calc_costs(data, embeddings, vocab, p=1, K_lda=70, n_words_keep=100, seed=1):
    topics, lda_centers, topic_proportions = fit_topics(data, embeddings, vocab, K_lda, seed)

    cost_embeddings = euclidean_distances(embeddings, embeddings) ** p
    cost_topics = np.zeros((topics.shape[0], topics.shape[0]))
    
    # Reduce topics to top-20 words
    if n_words_keep is not None:
        for k in range(K_lda):
            to_0_idx = np.argsort(-topics[k])[n_words_keep:]
            topics[k][to_0_idx] = 0

    for i in range(cost_topics.shape[0]):
        for j in range(i + 1, cost_topics.shape[0]):
            cost_topics[i, j] = sparse_ot(topics[i], topics[j], cost_embeddings)
    cost_topics = cost_topics + cost_topics.T

    out = {'X': data,
           'embeddings': embeddings,
           'topics': topics, 'proportions': topic_proportions,
           'cost_E': cost_embeddings, 'cost_T': cost_topics}

    return out


def wmd(p, q, C, truncate=None):
    """ Word mover's distance between distributions p and q with cost M."""
    if truncate is None:
        return sparse_ot(p, q, C)
    
    # Avoid changing p and q outside of this function
    p, q = np.copy(p), np.copy(q)
    
    to_0_p_idx = np.argsort(-p)[truncate:]
    p[to_0_p_idx] = 0
    to_0_q_idx = np.argsort(-q)[truncate:]
    q[to_0_q_idx] = 0
    
    return sparse_ot(p, q, C)


def rwmd(p, q, C):
    """ Relaxed WMD between distributions p and q with cost M."""
    active1 = np.where(p)[0]
    active2 = np.where(q)[0]
    C_reduced = C[active1][:, active2]
    l1 = (np.min(C_reduced, axis=1) * p[active1]).sum()
    l2 = (np.min(C_reduced, axis=0) * q[active2]).sum()
    return max(l1, l2)


def wcd(p, q, embeddings):
    """ Word centroid distance between p and q under embeddings."""
    m1 = np.mean(embeddings.T * p, axis=1)
    m2 = np.mean(embeddings.T * q, axis=1)
    return np.linalg.norm(m1 - m2)


def hott(p, q, C, threshold=None):
    """ Hierarchical optimal topic transport."""
    
    # Avoid changing p and q outside of this function
    p, q = np.copy(p), np.copy(q)
    
    k = len(p)
    if threshold is None:
        threshold = 1. / (k + 1)
        
    p[p < threshold] = 0
    q[q < threshold] = 0
    
    return sparse_ot(p, q, C)


def hoftt(p, q, C):
    """ Hierarchical optimal full topic transport."""
    return ot.emd2(p, q, C)


def calculate_hott_dist(i, j):
    return hott(topic_proportions[i], topic_proportions[j], C)


def calculate_hoftt_dist(i, j):
    return hoftt(topic_proportions[i], topic_proportions[j], C)


def knn_evaluation(X=None, y=None, distance_matrix=None, test_size=0.2, k=10, seed=1):
    """
    Split dataset and perform KNN classification, printing four evaluation metrics.
    
    Parameters:
    X -- Sample feature matrix (can be None if distance_matrix is provided)
    y -- Sample labels
    distance_matrix -- Precomputed distance matrix (can be None if X is provided)
    test_size -- Test set proportion
    k -- Number of neighbors in KNN
    
    Returns:
    None
    """
    if distance_matrix is None:
        if X is not None:
            distance_matrix = squareform(pdist(X, metric='cosine'))
        else:
            raise ValueError("Must provide X or distance_matrix")

    if y is None:
        raise ValueError("Must provide y")
        
    # Split dataset into training and test sets
    indices = np.arange(len(y))
    train_indices, test_indices, y_train, y_test = train_test_split(indices, y, test_size=test_size, random_state=seed)

    # Extract sub-matrix
    train_distance_matrix = distance_matrix[np.ix_(train_indices, train_indices)]
    test_distance_matrix = distance_matrix[np.ix_(test_indices, train_indices)]

    knn_model = KNeighborsClassifier(n_neighbors=k, metric='precomputed')

    # Train KNN using training distance sub-matrix (fit)
    knn_model.fit(train_distance_matrix, y_train)
    
    # Predict using test distance matrix
    y_pred = knn_model.predict(test_distance_matrix)
    
    # Evaluate classification model
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
    print(f"Accuracy: {accuracy:.6f}")
    print(f"Precision: {precision:.6f}")
    print(f"Recall: {recall:.6f}")
    print(f"F1-Score: {f1:.6f}")
    with open('0917_Aorta_metrics_results.csv', 'a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
    
        # Check if the file is empty; if so, write header row
        if file.tell() == 0:
            writer.writerow(['Seed', 'Accuracy', 'Precision', 'Recall', 'F1-Score'])
    
        # Write new row
        writer.writerow([seed, accuracy, precision, recall, f1])


def knn_evaluation_split(X=None, y=None, distance_matrix=None, train_size=0, k=10, seed=1):
    """
    Split dataset and perform KNN classification, printing four evaluation metrics.
    
    Parameters:
    X -- Sample feature matrix (can be None if distance_matrix is provided)
    y -- Sample labels
    distance_matrix -- Precomputed distance matrix (can be None if X is provided)
    train_size -- Training set size
    k -- Number of neighbors in KNN
    
    Returns:
    None
    """
    if distance_matrix is None:
        if X is not None:
            distance_matrix = squareform(pdist(X, metric='cosine'))
        else:
            raise ValueError("Must provide X or distance_matrix")

    if y is None:
        raise ValueError("Must provide y")

    # Split dataset into training and test sets
    train_indices = np.arange(train_size, dtype=int)
    test_indices = np.arange(train_size, len(y), dtype=int)
    y_train = [y[i] for i in train_indices]
    y_test = [y[i] for i in test_indices]

    # Extract sub-matrix
    train_distance_matrix = distance_matrix[np.ix_(train_indices, train_indices)]
    test_distance_matrix = distance_matrix[np.ix_(test_indices, train_indices)]

    knn_model = KNeighborsClassifier(n_neighbors=k, metric='precomputed')

    # Train KNN using training distance sub-matrix (fit)
    knn_model.fit(train_distance_matrix, y_train)
    
    # Predict using test distance matrix
    y_pred = knn_model.predict(test_distance_matrix)
    
    # Evaluate classification model
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
    print(f"Accuracy: {accuracy:.6f}")
    print(f"Precision: {precision:.6f}")
    print(f"Recall: {recall:.6f}")
    print(f"F1-Score: {f1:.6f}")
    with open('0917_Aorta_metrics_results.csv', 'a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
    
        # Check if the file is empty; if so, write header row
        if file.tell() == 0:
            writer.writerow(['Seed', 'Accuracy', 'Precision', 'Recall', 'F1-Score'])
    
        # Write new row
        writer.writerow([seed, accuracy, precision, recall, f1])


def generate_umap_visualizations(celltype, dist_matrix=None, data=None, n_neighbors=15, min_dist=0.1, embedding_suffix=None, fig_suffix=None):
    """
    Generate and save UMAP embedding image.
    
    Parameters:
    celltype -- List of cell types for color encoding
    dist_matrix -- Precomputed distance matrix (None if using raw data)
    data -- Raw data matrix (None if using precomputed distance matrix)
    n_neighbors -- UMAP parameter, number of neighbors
    min_dist -- UMAP parameter, minimum distance
    
    Returns:
    None
    """
    categories = pd.Categorical(celltype)
    print(categories.categories)
    integer_codes = categories.codes
    print(len(integer_codes))
    
    # Choose UMAP reducer based on whether precomputed distance matrix is provided
    if dist_matrix is not None:
        reducer = umap.UMAP(metric='precomputed')
        embedding = reducer.fit_transform(dist_matrix)
        
    elif data is not None:
        reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist)
        embedding = reducer.fit_transform(data)
    else:
        raise ValueError("Must provide dist_matrix or data")

    output_csv = f'{embedding_suffix}.csv'
    output_image = f'{fig_suffix}.png'
    
    # Save embedding results to CSV file
    embedding_df = pd.DataFrame(embedding, columns=['UMAP1', 'UMAP2'])
    embedding_df['celltype'] = categories
    embedding_df.to_csv(output_csv, index=False)
    
    cmap = plt.get_cmap('tab20')
    colors = cmap(np.arange(len(categories.categories)))
    color_dict = {categories.categories[i]: colors[i] for i in range(len(categories.categories))}
    cmap = ListedColormap([color_dict[ct] for ct in categories.categories])
    # Generate and save UMAP image
    plt.figure(figsize=(10, 7))
    sc = plt.scatter(embedding[:, 0], embedding[:, 1], c=integer_codes, s=0.1, cmap=cmap, alpha=0.8)
    plt.title('UMAP projection')
    plt.xlabel('UMAP 1')
    plt.ylabel('UMAP 2')

    handles = [mpatches.Patch(color=color_dict[ct], label=ct) for ct in categories.categories]
    plt.legend(handles=handles, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

    plt.savefig(output_image, dpi=300, bbox_inches='tight')
    plt.show()


# Cluster analysis function
def cluster_analysis(celltype, hott_matrix, mcd_matrix, expr_matrix, seed):
    # Get the number of cell types (to determine the number of clusters)
    n_clusters = len(set(celltype))
    
    # Initialize results dataframe
    df_results = pd.DataFrame(columns=['Seed', 
                                       'Silhouette Score (HOTT)', 'ARI (HOTT)',
                                       'Silhouette Score (MCD)', 'ARI (MCD)',
                                       'Silhouette Score (expr)', 'ARI (expr)'])
    
    # Store all distance matrices
    distance_matrices = [hott_matrix, mcd_matrix, expr_matrix]
    
    # Store result list, initially adding random seed
    results = [seed]  
    
    # Calculate clustering and evaluation results for each distance matrix
    for distance_matrix in distance_matrices:
        # Perform K-medoids clustering
        kmedoids = KMedoids(n_clusters=n_clusters, metric='precomputed', random_state=seed)
        labels = kmedoids.fit_predict(distance_matrix)

        # Calculate silhouette score
        silhouette_avg = silhouette_score(distance_matrix, labels, metric='precomputed')

        # Calculate adjusted Rand index (assuming 'celltype' is the ground truth labels)
        ari_score = adjusted_rand_score(celltype, labels)

        # Store results
        results.extend([silhouette_avg, ari_score])

    # Add results to dataframe
    df_results.loc[len(df_results)] = results
    
    # Check if file exists
    file_exists = os.path.isfile('clustering_results.csv')
    
    # Save results to file
    df_results.to_csv('clustering_results.csv', index=False, mode='a', header=not file_exists)
