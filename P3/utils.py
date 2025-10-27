import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


def read_network(filepath: str) -> nx.Graph:
    graph = nx.Graph()
    with open(filepath) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                graph.add_edge(parts[0], parts[1], weight=1)
            else:
                graph.add_edge(parts[0], parts[1], weight=float(parts[2]))
    return graph


def draw_network(G, title, n=50, pos=None):
    degree_dict = dict(G.degree())
    if n is None:
        n = len(degree_dict)
    top_nodes = sorted(degree_dict, key=degree_dict.get, reverse=True)[:n]
    node_sizes = [degree_dict[node] * 50 for node in G.nodes()]

    clusters = list(nx.algorithms.community.greedy_modularity_communities(G, weight='Weight'))
    print(f"# clusters: {len(clusters)}")
    cluster_dict = {}
    for i, cluster in enumerate(clusters):
        for node in cluster:
            cluster_dict[node] = i

    cmap = plt.colormaps['Set3'].resampled(len(clusters))
    node_colors = [cmap(cluster_dict[node]) for node in G.nodes()]

    if pos is None:
        pos = nx.spring_layout(G, seed=0, k=1.0, scale=3)
    width = np.array(list(nx.get_edge_attributes(G, "weight").values()))
    if width is not None and width.size > 0:
        norm_width = (width - np.min(width) + 1) / np.max(width) * 2
    else:
        norm_width = 1

    plt.figure(figsize=(12, 12))
    nx.draw_networkx_edges(G, pos, alpha=0.4, width=norm_width)
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, edgecolors='black')

    labels = {node: node for node in top_nodes}
    nx.draw_networkx_labels(G, pos, labels, font_size=8, font_color='black')
    if n < len(degree_dict):
        title = f"{title} (Top {n} labeled)"
    plt.title(title, fontsize=14)
    plt.axis('off')
    plt.show()

