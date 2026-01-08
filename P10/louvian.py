import csv
import random
from collections import defaultdict

import networkx as nx
import numpy as np
from matplotlib import pyplot as plt

np.random.seed(4242)


def modularity(G, node2comm, weight="weight"):
    """
    compute modularity Q for partition node2comm
    """
    m = G.size(weight=weight)
    if m == 0:
        return 0

    # sum of degrees per community
    comm_tot = defaultdict(float)
    # sum of internal edges per community
    comm_in = defaultdict(float)

    for u in G.nodes():
        cu = node2comm[u]
        for v, data in G[u].items():
            w = data.get(weight, 1.0)
            if node2comm[v] == cu:
                comm_in[cu] += w
            comm_tot[cu] += w

    Q = 0.0
    for c in comm_tot:
        Q += (comm_in[c] / (2 * m)) - (comm_tot[c] / (2 * m)) ** 2

    return Q


def movenodes(G, node2comm, weight="weight"):
    """
    determine the best comm for each node (in random order)
    """
    # build comm2nodes
    comm_members = defaultdict(set)
    for v, c in node2comm.items():
        comm_members[c].add(v)

    improved = True
    while improved:
        improved = False
        Hold = modularity(G, node2comm, weight=weight)

        nodes = list(G.nodes())
        random.shuffle(nodes)

        for v in nodes:
            current_c = node2comm[v]

            # remove v from its community
            comm_members[current_c].remove(v)
            if len(comm_members[current_c]) == 0:
                del comm_members[current_c]

            # compute modularity gain for each neighbor community
            neighbor_comms = set(node2comm[u] for u in G[v])
            best_delta = 0
            best_comm = current_c

            for c in neighbor_comms:
                # try placing v into community c
                node2comm[v] = c
                comm_members[c].add(v)

                newQ = modularity(G, node2comm, weight=weight)

                # undo
                comm_members[c].remove(v)
                node2comm[v] = current_c

                delta = newQ - Hold
                if delta > best_delta:
                    best_delta = delta
                    best_comm = c

            # move v if improvement
            if best_comm != current_c:
                node2comm[v] = best_comm
                comm_members[best_comm].add(v)
                improved = True
            else:
                # put v back
                comm_members[current_c].add(v)

    return node2comm


def aggregategraph(G, node2comm, weight="weight"):
    """
    communities become nodes and edges between communities sum weights
    """
    H = nx.Graph()

    # community nodes
    for c in set(node2comm.values()):
        H.add_node(c)

    # edges between communities
    for u, v, data in G.edges(data=True):
        cu = node2comm[u]
        cv = node2comm[v]
        w = data.get(weight, 1.0)

        if H.has_edge(cu, cv):
            H[cu][cv][weight] += w
        else:
            H.add_edge(cu, cv, **{weight: w})

    return H


def singletonpartition(G):
    return {v: v for v in G.nodes()}


def flatten_partitions(level_partitions):
    """
    level_partitions: [part0, part1, ..., partL]
    Returns:
      final mapping from original nodes (nodes of G0) -> list of community ids that it's been assigned to at each level.
    """
    final_mapping = {}

    for k in level_partitions[0].keys():
        final_mapping[k] = [level_partitions[0][k]]

    for node in final_mapping.keys():
        for level in range(1, len(level_partitions) - 1):
            part = level_partitions[level]
            final_mapping[node].append(part[final_mapping[node][-1]])

    return final_mapping


def louvain(G, weight="weight", max_iter=50):
    """
    top-level louvain
    """
    # init: each node in its own community
    level_partitions = []
    node2comm = singletonpartition(G)

    for i in range(max_iter):
        # 1: local moving
        node2comm = movenodes(G, node2comm, weight=weight)
        level_partitions.append(node2comm.copy())

        # check termination: each community has size 1
        comm_sizes = defaultdict(int)
        for c in node2comm.values():
            comm_sizes[c] += 1

        done = all(size == 1 for size in comm_sizes.values())
        if done:
            break

        # 2: aggregate graph
        G = aggregategraph(G, node2comm, weight=weight)

        # 3: reset partition to singleton
        node2comm = singletonpartition(G)

    return level_partitions


def normalize_partition(node2comm):
    """
    Remap community IDs to 0..K-1
    """
    unique_comms = sorted(set(node2comm.values()))
    mapping = {old: new for new, old in enumerate(unique_comms)}
    return {node: mapping[cid] for node, cid in node2comm.items()}


def visualize_partition_clustered(G, node2comm, node_size=300, title=""):
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
            pos[node] = np.array([x + offset, y + 2 if i % 2 == 0 else y])

        offset += 2  # spacing

    # colors
    colors = [cmap(node2comm[n]) for n in G.nodes()]

    plt.figure(figsize=(10, 7))
    nx.draw(G, pos, node_color=colors, with_labels=True, node_size=node_size, edge_color="gray")

    plt.title(f"Louvain Communities: {title}")
    plt.axis("off")
    plt.show()


def save_node2comm_to_csv(node2comm, path):
    node2comm = normalize_partition(node2comm)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        for node in sorted(node2comm.keys()):
            writer.writerow([node + 1, node2comm[node] + 1])
