# -*- coding: utf-8 -*-
import os
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd

def load_graph():
    """
    Load the graph data: prefer GraphML if available; otherwise, build from edge list and node attributes.
    """
    graphml_path = "example_graph.graphml"
    edgelist_path = "example_graph_edgelist.txt"
    node_ops_path = "example_node_operations.txt"

    if os.path.exists(graphml_path):
        G = nx.read_graphml(graphml_path)
        print(f"Loaded graph from '{graphml_path}'.")
    else:
        if os.path.exists(edgelist_path) and os.path.exists(node_ops_path):
            G = nx.DiGraph()
            node_ops = {}
            with open(node_ops_path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split("\t")
                    if not parts: continue
                    name = parts[0]
                    op = parts[1] if len(parts) > 1 else ""
                    node_ops[name] = op
                    G.add_node(name, operation=op)
            with open(edgelist_path, "r", encoding="utf-8") as f:
                for line in f:
                    u, v = line.strip().split()
                    if not G.has_node(u):
                        G.add_node(u, operation=node_ops.get(u, "leaf"))
                    if not G.has_node(v):
                        G.add_node(v, operation=node_ops.get(v, "leaf"))
                    G.add_edge(u, v)
            print(f"Rebuilt graph from '{edgelist_path}' + '{node_ops_path}'.")
        else:
            raise FileNotFoundError("Graph files not found. Please check the path.")
    return G

def read_node_features(xlsx_path: str):
    """
    Read node features from the first sheet of an Excel file. Return a dictionary.
    """
    df = pd.read_excel(xlsx_path, sheet_name=0)
    node_col, feat_col = df.columns[0], df.columns[1]
    return {
        str(row[node_col]).strip(): float(row[feat_col]) if pd.notna(row[feat_col]) else float("nan")
        for _, row in df.iterrows()
    }

def initial_leaf_prune(G: nx.DiGraph, features: dict):
    """
    Remove leaf nodes that are not included in the feature dictionary.
    """
    G0 = G.copy()
    to_remove = [n for n, d in G.nodes(data=True)
                 if d.get("operation", "") == "leaf" and n not in features]
    if to_remove:
        G0.remove_nodes_from(to_remove)
        print(f"Removed {len(to_remove)} leaf nodes.")
    else:
        print("No leaf nodes removed.")
    return G0

def prune_graph_by_threshold(G, features, threshold, special_node):
    """
    Prune the graph based on feature threshold and a designated special node. 
    Ensure key nodes remain connected.
    """
    if special_node not in features or abs(features[special_node] - 1.0) > 1e-9:
        raise ValueError(f"Special node '{special_node}' not found or has feature value not equal to 1.")

    S = {n for n, v in features.items() if not pd.isna(v) and v > threshold}
    S.add(special_node)
    H_undir = G.subgraph(S).to_undirected()

    T = set()
    for n in S:
        if n == special_node: continue
        if nx.has_path(H_undir, special_node, n):
            continue
        try:
            path = nx.shortest_path(G.to_undirected(), source=special_node, target=n)
            T.update(p for p in path if p not in S)
        except nx.NetworkXNoPath:
            continue

    keep_nodes = S.union(T)
    return G.subgraph(keep_nodes).copy(), S, T

def visualize_and_save_pruned(G, S, T, output_img="pruned_graph.png"):
    """
    Visualize and save the pruned graph, color nodes by type.
    """
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G, k=0.5, iterations=50)
    colors = [
        "#A0CBE2" if n in S else "#FF8C00" if n in T else "#D3D3D3"
        for n in G.nodes()
    ]
    nx.draw_networkx_nodes(G, pos, node_size=700, node_color=colors)
    nx.draw_networkx_edges(G, pos, arrowstyle="->", arrowsize=12, edge_color="#555555")
    nx.draw_networkx_labels(G, pos, font_size=9)

    plt.title("Pruned Variable Influence Graph")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_img, dpi=300)
    print(f"Saved graph image to '{output_img}'.")
    plt.show()

def save_pruned_graph_files(G, prefix="pruned_graph"):
    """
    Save the pruned graph in GraphML format, and export edge list and node attributes.
    """
    nx.write_graphml(G, f"{prefix}.graphml")
    nx.write_edgelist(G, f"{prefix}_edgelist.txt", data=False)
    with open(f"{prefix}_node_operations.txt", "w", encoding="utf-8") as f:
        for n, d in G.nodes(data=True):
            f.write(f"{n}\t{d.get('operation','')}\n")
    print(f"Graph saved to '{prefix}.*'.")

if __name__ == "__main__":
    G_original = load_graph()

    xlsx_path = "node_features.xlsx"
    if not os.path.exists(xlsx_path):
        raise FileNotFoundError(f"Missing feature file: '{xlsx_path}'")
    features = read_node_features(xlsx_path)

    special_nodes = [n for n, v in features.items() if abs(v - 1.0) < 1e-9]
    if not special_nodes:
        raise ValueError("No special node with feature value 1 found.")
    if len(special_nodes) > 1:
        print(f"Multiple special nodes found, using default: '{special_nodes[0]}'")
    special_node = special_nodes[0]

    G0 = initial_leaf_prune(G_original, features)

    while True:
        try:
            threshold = float(input("Enter threshold (e.g. 0.3): "))
            break
        except ValueError:
            print("Please enter a valid number.")

    G_pruned, S, T = prune_graph_by_threshold(G0, features, threshold, special_node)
    print(f"Threshold = {threshold}, |S| = {len(S)}, |T| = {len(T)}, final graph has {G_pruned.number_of_nodes()} nodes and {G_pruned.number_of_edges()} edges.")

    visualize_and_save_pruned(G_pruned, S, T)
    save_pruned_graph_files(G_pruned)
