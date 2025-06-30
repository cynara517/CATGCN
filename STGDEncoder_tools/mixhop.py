# -*- coding: utf-8 -*-
import os
import glob
import networkx as nx
import numpy as np
import pandas as pd

def load_node_feature_table(feature_file: str):
    """
    Load feature values for leaf nodes from CSV or Excel.
    Format: first column = node name, second column = value.
    """
    name, ext = os.path.splitext(feature_file)
    if ext.lower() == ".csv":
        df = pd.read_csv(feature_file)
    else:
        df = pd.read_excel(feature_file, sheet_name=0)
    node_col, feat_col = df.columns[0], df.columns[1]
    features = {}
    for _, row in df.iterrows():
        n = str(row[node_col]).strip()
        try:
            v = float(row[feat_col])
        except:
            v = np.nan
        features[n] = v
    return features

def process_subgraph_corrected(subgraph_path: str, features: dict, epsilon: float = 1e-6):
    """
    Perform MixHop-style aggregation on a directed subgraph ending at special node Z.

    Returns a dictionary containing:
        - 'subgraph': filename
        - 'source': inferred from filename
        - 'special': target node (Z)
        - 'H': max hop count
        - 'y_h_list': list of hop-level aggregates
        - 'sum_y': sum of y_h_list
        - 'node_list': ordered list of nodes
        - 'operation_map': {node: operation}
        - 'x_initial': initial feature vector
        - 'dist_map': {node: hop distance to Z}
    """
    G = nx.read_graphml(subgraph_path)
    if not isinstance(G, nx.DiGraph):
        G = G.to_directed()

    fname = os.path.basename(subgraph_path)

    # 1) Locate special node (Z)
    special_nodes = [n for n, d in G.nodes(data=True) if d.get("operation") == "special"]
    if len(special_nodes) != 1:
        raise RuntimeError(f"Subgraph '{fname}' must contain exactly one node with operation='special'. Found {len(special_nodes)}.")
    special = special_nodes[0]

    # 2) Parse source node from filename
    parts = fname.split('.')[0].split('_to_')
    source = parts[0].replace("subgraph_", "")

    # 3) Node list & operation map
    node_list = list(G.nodes())
    operation_map = {n: G.nodes[n].get("operation", "") for n in node_list}
    idx = {n: i for i, n in enumerate(node_list)}
    N = len(node_list)

    # 4) Build initial feature vector
    x_initial = np.zeros(N, dtype=float)
    for i, n in enumerate(node_list):
        op = operation_map[n]
        if n == special:
            x_initial[i] = 1.0
        elif op == "leaf":
            v = features.get(n, np.nan)
            x_initial[i] = 0.0 if np.isnan(v) else v
        else:
            x_initial[i] = epsilon  # small value for intermediate nodes

    # 5) Compute directed distance to Z using BFS on reversed graph
    G_rev = G.reverse(copy=False)
    dist_map = {special: 0}
    queue = [special]
    while queue:
        u = queue.pop(0)
        for v in G_rev.neighbors(u):
            if v not in dist_map:
                dist_map[v] = dist_map[u] + 1
                queue.append(v)

    # 6) Maximum hop H
    H = max(dist_map.values()) if dist_map else 0

    # 7) Compute y_h = sum of x_initial[n] for nodes with distance h
    y_h_list = []
    for h in range(1, H + 1):
        sum_val = sum(
            x_initial[idx[n]]
            for n, d in dist_map.items()
            if d == h
        )
        y_h_list.append(sum_val)
    sum_y = float(sum(y_h_list))

    return {
        "subgraph": fname,
        "source": source,
        "special": special,
        "H": H,
        "y_h_list": y_h_list,
        "sum_y": sum_y,
        "node_list": node_list,
        "operation_map": operation_map,
        "x_initial": x_initial.tolist(),
        "dist_map": dist_map
    }

def main():
    # 1) Load feature table
    feature_file = "example_node_features.xlsx"
    if not os.path.exists(feature_file):
        raise FileNotFoundError(f"Feature file not found: '{feature_file}'")
    features = load_node_feature_table(feature_file)
    print(f"Loaded features for {len(features)} nodes from '{feature_file}'.")

    # 2) Find subgraph files
    subgraphs = sorted(glob.glob("subgraph_*_to_*.graphml"))
    if not subgraphs:
        raise FileNotFoundError("No subgraph files matching 'subgraph_*_to_*.graphml' found in current directory.")

    summary_records = []

    for path in subgraphs:
        print(f"Processing '{path}' …")
        res = process_subgraph_corrected(path, features, epsilon=1e-6)

        # 3) Export per-subgraph full detail
        fname = res["subgraph"]
        node_list = res["node_list"]
        operation_map = res["operation_map"]
        x_initial = res["x_initial"]
        dist_map = res["dist_map"]

        rows = []
        for n in node_list:
            rows.append({
                "Node": n,
                "operation": operation_map[n],
                "x_initial": x_initial[node_list.index(n)],
                "dist_to_Z": dist_map.get(n, None)
            })
        df_details = pd.DataFrame(rows)
        details_csv = f"mixhop_full_details_{fname.replace('.graphml', '')}.csv"
        df_details.to_csv(details_csv, index=False, encoding="utf-8-sig")
        print(f"  → Saved node-level details to '{details_csv}'")

        # 4) Record summary
        record = {
            "Subgraph File": fname,
            "Source Node": res["source"],
            "Special Node": res["special"],
            "Max Hop H": res["H"],
            "Sum_{h=1..H}": res["sum_y"]
        }
        for h_idx, yh in enumerate(res["y_h_list"], start=1):
            record[f"y_hop_{h_idx}"] = yh
        summary_records.append(record)

    # 5) Save summary CSV
    df_summary = pd.DataFrame(summary_records)
    summary_csv = "mixhop_summary.csv"
    df_summary.to_csv(summary_csv, index=False, encoding="utf-8-sig")
    print(f"\nSaved summary to '{summary_csv}':")
    print(df_summary)

if __name__ == "__main__":
    main()
