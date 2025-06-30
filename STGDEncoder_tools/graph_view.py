# -*- coding: utf-8 -*-
import os
import networkx as nx
import matplotlib.pyplot as plt

# 1. Try loading an existing GraphML file
graphml_path = "variable_influence_graph.graphml"
edgelist_path = "variable_influence_edgelist.txt"
node_ops_path = "node_operations.txt"

if os.path.exists(graphml_path):
    # Load graph from GraphML if available
    G = nx.read_graphml(graphml_path)
    print(f"Loaded graph from '{graphml_path}'.")
else:
    # Otherwise, rebuild from edgelist + node operations
    if os.path.exists(edgelist_path) and os.path.exists(node_ops_path):
        G = nx.DiGraph()

        # Load node attributes
        node_ops = {}
        with open(node_ops_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("\t")
                name, op = parts[0], parts[1] if len(parts) >= 2 else ""
                node_ops[name] = op
                G.add_node(name, operation=op)

        # Load edge list
        with open(edgelist_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                u, v = line.split()
                if not G.has_node(u):
                    G.add_node(u, operation=node_ops.get(u, "leaf"))
                if not G.has_node(v):
                    G.add_node(v, operation=node_ops.get(v, "leaf"))
                G.add_edge(u, v)

        print(f"Rebuilt graph from '{edgelist_path}' + '{node_ops_path}'.")
    else:
        # If both sources are missing, raise error
        raise FileNotFoundError(
            "No valid graph files found. Please make sure either "
            "'variable_influence_graph.graphml' exists, or both "
            "'variable_influence_edgelist.txt' and 'node_operations.txt' are available."
        )

# 2. Visualize the graph using spring layout
plt.figure(figsize=(12, 8))
pos = nx.spring_layout(G, k=0.5, iterations=50)

# Node coloring: gold for leaf, light blue for others
colors = []
for _, data in G.nodes(data=True):
    op = data.get("operation", "")
    if op == "leaf":
        colors.append("#FFD700")   # gold
    else:
        colors.append("#A0CBE2")   # light blue

# Draw nodes, edges, and labels
nx.draw_networkx_nodes(G, pos, node_size=900, node_color=colors)
nx.draw_networkx_edges(G, pos, arrowstyle="->", arrowsize=15, edge_color="#333333")
nx.draw_networkx_labels(G, pos, font_size=10)

plt.title("Variable Influence Graph Visualization")
plt.axis("off")
plt.tight_layout()

# 3. Save graph as PNG
output_img = "variable_influence_graph.png"
plt.savefig(output_img, dpi=300)
print(f"Graph figure saved as '{output_img}'.")

# 4. Optionally display the plot
plt.show()
