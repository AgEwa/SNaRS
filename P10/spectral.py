import csv
import time

import numpy as np
import pandas as pd
from numpy.linalg import eigh
from sklearn.cluster import KMeans
import networkx as nx
import matplotlib.pyplot as plt

import os
os.environ["LOKY_MAX_CPU_COUNT"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

np.random.seed(0)

def read_adjacency_from_csv(path, has_header=False):
    header = 0 if has_header else None
    df = pd.read_csv(path, header=header)
    A = df.to_numpy(dtype=float)
    return A


def spectral_communities_ng(A, k, eps=1e-12, random_state=4242, n_init=20, max_iter=300):
    """
    Ng–Jordan–Weiss spectral clustering on adjacency matrix A.
    """
    A = np.asarray(A, dtype=float)
    assert A.shape[0] == A.shape[1], "A must be square"

    # degree vector and normalized affinity matrix L
    d = A.sum(axis=1)
    d_safe = np.where(d > 0, d, eps)
    D_inv_sqrt = 1.0 / np.sqrt(d_safe)
    L = A * D_inv_sqrt[:, None] * D_inv_sqrt[None, :]

    # eigen-decomposition
    evals, evecs = eigh(L)
    idx = np.argsort(evals)[::-1]  # descending
    idx_k = idx[:k]
    X = evecs[:, idx_k]

    # row-normalize X to get Y
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.where(norms > 0, norms, eps)
    Y = X / norms

    # K-means on rows of Y
    km = KMeans(n_clusters=k, n_init=n_init, max_iter=max_iter, random_state=random_state)
    labels = km.fit_predict(Y)

    return labels, X


def estimate_k_from_spectrum(A, k_max=10, eps=1e-12):
    """
    estimate number of communities using eigengap on L = D^{-1/2} A D^{-1/2}.
    """
    A = np.asarray(A, dtype=float)
    n = A.shape[0]
    assert A.shape[0] == A.shape[1], "A must be square"

    d = A.sum(axis=1)
    d_safe = np.where(d > 0, d, eps)
    D_inv_sqrt = 1.0 / np.sqrt(d_safe)
    L = A * D_inv_sqrt[:, None] * D_inv_sqrt[None, :]

    evals, _ = eigh(L)
    evals_desc = np.sort(evals)[::-1]

    m = min(n, k_max + 1)
    evals_desc = evals_desc[:m]

    if len(evals_desc) == 1:
        return 1, evals_desc

    gaps = evals_desc[:-1] - evals_desc[1:]
    k_hat = int(np.argmax(gaps) + 1)
    k_hat = max(1, min(k_hat, k_max))

    return k_hat


def spectral_communities_ng_auto_k(A, k_max=10, eps=1e-12, random_state=4242):
    """
    Ng–Jordan–Weiss spectral clustering with automatic k selection
    """
    k_hat = estimate_k_from_spectrum(A, k_max=k_max, eps=eps)
    labels, X = spectral_communities_ng(A, k=k_hat, eps=eps, random_state=random_state)
    return labels, k_hat, X


def detect_communities_from_csv(path, k=None, k_max=10, has_header=False, random_state=4242):
    """
    csv -> community labels pipeline
    """
    A = read_adjacency_from_csv(path, has_header=has_header)

    if k is None:
        start = time.perf_counter()
        labels, k_hat, X = spectral_communities_ng_auto_k(A, k_max=k_max, random_state=random_state)
        end = time.perf_counter()
        return A, labels, k_hat, end-start
    else:
        start = time.perf_counter()
        labels, X = spectral_communities_ng(A, k=k, random_state=random_state)
        end = time.perf_counter()
        return A, labels, k, end-start



def normalize_partition(labels):
    """
    convert cluster labels into a dict {node: normalized_cluster_id}
    """
    labels = np.asarray(labels)
    unique = {c: i for i, c in enumerate(np.unique(labels))}
    return {i: unique[labels[i]] for i in range(len(labels))}


def visualize_partition_clustered(A_or_G, node2comm, node_size=300, title=""):
    # Build graph if adjacency matrix is given
    if isinstance(A_or_G, np.ndarray):
        G = nx.from_numpy_array(A_or_G)
    else:
        G = A_or_G

    node2comm = normalize_partition(node2comm)

    # build comm2nodes
    comm2nodes = {}
    for node, cid in node2comm.items():
        comm2nodes.setdefault(cid, set()).add(node)

    cmap = plt.cm.get_cmap("tab20", len(comm2nodes))

    pos = {}
    offset = 0

    # layout per community and shift them
    for i, (cid, members) in enumerate(comm2nodes.items()):
        subG = G.subgraph(members)

        sub_pos = nx.spring_layout(subG, seed=42)

        for node, (x, y) in sub_pos.items():
            pos[node] = np.array([x + offset, y+2 if i%2==0 else y])

        offset += 2  # spacing

    # colors
    colors = [cmap(node2comm[n]) for n in G.nodes()]

    plt.figure(figsize=(10, 7))
    nx.draw(
        G,
        pos,
        node_color=colors,
        with_labels=True,
        node_size=node_size,
        edge_color="gray"
    )

    plt.title(title)
    plt.axis("off")
    plt.show()

def save_node2comm_to_csv(node2comm, path):
    node2comm = normalize_partition(node2comm)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        for node in sorted(node2comm.keys()):
            writer.writerow([node+1, node2comm[node]+1])
