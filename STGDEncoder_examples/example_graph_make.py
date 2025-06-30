import networkx as nx
import matplotlib.pyplot as plt

# Create a sample directed graph
G_test = nx.DiGraph()

# Add nodes with 'operation' attribute: leaf, med, special
G_test.add_node('Z', operation='special')  # target node

# Add leaf nodes
for node in ['A', 'B', 'C', 'D', 'E', 'F']:
    G_test.add_node(node, operation='leaf')

# Add intermediate nodes
for node in ['M1', 'M2', 'M3', 'M4']:
    G_test.add_node(node, operation='med')

# Add edges (example paths)
G_test.add_edge('A', 'M1')
G_test.add_edge('M1', 'Z')

G_test.add_edge('B', 'M2')
G_test.add_edge('M2', 'M3')
G_test.add_edge('M3', 'Z')

G_test.add_edge('C', 'Z')

G_test.add_edge('D', 'M4')
G_test.add_edge('M4', 'M3')

G_test.add_edge('E', 'F')  # disconnected from Z

# Visualize and save graph as PNG
plt.figure(figsize=(8, 6))
pos = nx.spring_layout(G_test, seed=42)

colors = []
for n, data in G_test.nodes(data=True):
    op = data.get('operation')
    if op == 'special':
        colors.append('#27ae60')  # green: target
    elif op == 'leaf':
        colors.append('#e74c3c')  # red: leaf
    else:
        colors.append('#95a5a6')  # gray: intermediate

nx.draw_networkx_nodes(G_test, pos, node_size=600, node_color=colors)
nx.draw_networkx_edges(G_test, pos, arrowstyle='->', arrowsize=12, edge_color='#2c3e50')
nx.draw_networkx_labels(G_test, pos, font_size=10)
plt.title("Sample Directed Graph")
plt.axis("off")
plt.tight_layout()
plt.savefig("example_graph.png", dpi=300)
plt.close()

# Export as GraphML
nx.write_graphml(G_test, "example_graph.graphml")

# Export edge list
nx.write_edgelist(G_test, "example_graph_edgelist.txt", data=False)

# Export node attributes
with open("example_node_operations.txt", "w", encoding="utf-8") as f:
    for n, data in G_test.nodes(data=True):
        f.write(f"{n}\t{data.get('operation', '')}\n")

print("Generated and saved example graph:")
print(" - example_graph.png")
print(" - example_graph.graphml")
print(" - example_graph_edgelist.txt")
print(" - example_node_operations.txt")
