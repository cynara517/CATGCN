# -*- coding: utf-8 -*-
import os
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd

def load_graph():
    """
    Load graph from 'example_graph.graphml' if available;
    otherwise, rebuild from 'example_graph_edgelist.txt' and 'example_node_operations.txt'.
    Returns a NetworkX directed graph G (each node has an 'operation' attribute).
    """
    graphml_path = "example_graph.graphml"
    edgelist_path = "example_graph_edgelist.txt"
    node_ops_path = "example_node_operations.txt"

    if os.path.exists(graphml_path):
        G = nx.read_graphml(graphml_path)
        print(f"Loaded graph from '{graphml_path}'.")
    elif os.path.exists(edgelist_path) and os.path.exists(node_ops_path):
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
        print(f"Rebuilt graph from '{edgelist_path}' and '{node_ops_path}'.")
    else:
        raise FileNotFoundError("Graph files not found.")
    return G

def read_node_features(xlsx_path: str):
    """
    Read node features from Excel file (default sheet 0).
    Returns a dictionary {node_name: feature_value}.
    """
    df = pd.read_excel(xlsx_path, sheet_name=0)
    node_col, feat_col = df.columns[0], df.columns[1]
    features = {}
    for _, row in df.iterrows():
        name = str(row[node_col]).strip()
        try:
            val = float(row[feat_col])
        except:
            val = float("nan")
        features[name] = val
    return features

def extract_subgraph_for_node(G: nx.DiGraph, source_node: str, special_node: str):
    """
    Extract all simple paths from source_node to special_node in G.
    Returns (G_sub, all_paths).
    """
    G_sub = nx.DiGraph()
    all_paths = []
    if source_node not in G or special_node not in G:
        return G_sub, all_paths
    try:
        paths = list(nx.all_simple_paths(G, source=source_node, target=special_node))
    except nx.NetworkXNoPath:
        paths = []
    for path in paths:
        all_paths.append(path)
        for u, v in zip(path[:-1], path[1:]):
            if not G_sub.has_node(u):
                G_sub.add_node(u, operation=G.nodes[u].get("operation", ""))
            if not G_sub.has_node(v):
                G_sub.add_node(v, operation=G.nodes[v].get("operation", ""))
            if not G_sub.has_edge(u, v):
                G_sub.add_edge(u, v)
    return G_sub, all_paths

def visualize_single_subgraph(G_sub, source_node, special_node, save_path):
    """
    Visualize a subgraph with color:
    - red for source_node
    - green for special_node
    - gray for others
    Save to save_path.
    """
    plt.figure(figsize=(8, 6))
    pos = nx.spring_layout(G_sub, k=0.5, iterations=50)
    node_colors = [
        "#e74c3c" if n == source_node else
        "#27ae60" if n == special_node else
        "#95a5a6" for n in G_sub.nodes()
    ]
    nx.draw_networkx_nodes(G_sub, pos, node_size=600, node_color=node_colors)
    nx.draw_networkx_edges(G_sub, pos, arrowstyle="->", arrowsize=12, edge_color="#2c3e50")
    nx.draw_networkx_labels(G_sub, pos, font_size=9)
    plt.title(f"Subgraph: {source_node} → {special_node}")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"Saved visualization: {save_path}")
    plt.show()

def save_subgraph_files(G_sub, prefix):
    """
    Save G_sub to:
    - GraphML
    - edgelist
    - node operation list
    """
    nx.write_graphml(G_sub, f"{prefix}.graphml")
    nx.write_edgelist(G_sub, f"{prefix}_edgelist.txt", data=False)
    with open(f"{prefix}_node_ops.txt", "w", encoding="utf-8") as f:
        for n, d in G_sub.nodes(data=True):
            f.write(f"{n}\t{d.get('operation','')}\n")
    print(f"Saved: '{prefix}.graphml', '{prefix}_edgelist.txt', '{prefix}_node_ops.txt'")

if __name__ == "__main__":
    G = load_graph()
    xlsx_path = "example_node_features.xlsx"
    if not os.path.exists(xlsx_path):
        raise FileNotFoundError(f"Missing: '{xlsx_path}'")
    features = read_node_features(xlsx_path)
    special_nodes = [n for n, v in features.items() if abs(v - 1.0) < 1e-9]
    if not special_nodes:
        raise ValueError("No node with feature value == 1 found.")
    if len(special_nodes) > 1:
        print(f"Multiple special nodes found. Using: '{special_nodes[0]}'")
    special_node = special_nodes[0]
    normal_nodes = [n for n in features if n != special_node]
    for source_node in normal_nodes:
        print(f"\nProcessing: '{source_node}' → '{special_node}'")
        G_sub, all_paths = extract_subgraph_for_node(G, source_node, special_node)
        if not all_paths:
            print(f"  No path found. Skipping.")
            continue
        print(f"  {len(all_paths)} path(s) found:")
        for i, path in enumerate(all_paths, 1):
            print(f"    Path {i}: {' -> '.join(path)}")
        prefix = f"subgraph_{source_node.replace('/', '_').replace(' ', '_')}_to_{special_node}"
        visualize_single_subgraph(G_sub, source_node, special_node, save_path=prefix + ".png")
        save_subgraph_files(G_sub, prefix=prefix)
    print("\nAll done.")