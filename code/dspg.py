class Graph:
    def __init__(self):
        self.graph = {}
        
    def add_edge(self, src, tgt):
        if src not in self.graph:
            self.graph[src] = []
        self.graph[src].append(tgt)
        
    def get_all_paths(self, start, end, path=[]):
        path = path + [start]
        if start == end:
            return [path]
        if start not in self.graph:
            return []
        paths = []
        for node in self.graph[start]:
            if node not in path:
                new_paths = self.get_all_paths(node, end, path)
                for p in new_paths:
                    paths.append(p)
        return paths

class DSPGChecker:
    @staticmethod
    def check(nodes_to_pnps, src, tgt, graph):
        # Check if a path exists from src to tgt
        paths = graph.get_all_paths(src, tgt)
        if not paths:
            return False, []
        
        # For each path, check intermediate node conditions
        intermediate_nodes = []
        for path in paths:
            intermediate = path[1:-1]
            if (src in nodes_to_pnps and tgt not in nodes_to_pnps[src]) or \
               (tgt in nodes_to_pnps and src not in nodes_to_pnps[tgt]):
                return False, []
            intermediate_nodes.extend(intermediate)
        
        return True, intermediate_nodes

class DSPGSynthesizer:
    def __init__(self, graph):
        self.graph = graph
        self.nodes_to_pnps = {}
        self.synthesized_graph = Graph()
        
    def synthesize(self):
        # Start with paths to process from each node
        for node in self.graph.graph:
            for target in self.graph.graph:
                if node != target:
                    paths = self.graph.get_all_paths(node, target)
                    for path in paths:
                        # Check the DSPG conditions for each path
                        is_valid, intermediate_nodes = DSPGChecker.check(self.nodes_to_pnps, node, target, self.graph)
                        if is_valid:
                            # Add valid path to synthesized graph
                            for i in range(len(path) - 1):
                                self.synthesized_graph.add_edge(path[i], path[i+1])
                            for node in intermediate_nodes:
                                if node not in self.nodes_to_pnps:
                                    self.nodes_to_pnps[node] = []
                                self.nodes_to_pnps[node].append((node, target))
        
        # Return the synthesized graph
        return self.synthesized_graph


# Example Usage:

# Graph 1 (from first image)
g1 = Graph()
g1.add_edge('A', 'B')
g1.add_edge('B', 'C')
g1.add_edge('C', 'E')
g1.add_edge('A', 'D')
g1.add_edge('D', 'E')
g1.add_edge('B', 'E')

# Graph 2 (from second image)
g2 = Graph()
g2.add_edge('A', 'B')
g2.add_edge('B', 'C')
g2.add_edge('B', 'D')
g2.add_edge('C', 'D')

# Create DSPG synthesizers for both graphs
synthesizer1 = DSPGSynthesizer(g1)
synthesizer2 = DSPGSynthesizer(g2)

# Synthesize the DSPGs and get the synthesized graph
synthesized_graph1 = synthesizer1.synthesize()
synthesized_graph2 = synthesizer2.synthesize()

# Print the synthesized graph edges for both graphs
print("Synthesized Graph 1:")
for src in synthesized_graph1.graph:
    for tgt in synthesized_graph1.graph[src]:
        print(f"{src} -> {tgt}")

print("\nSynthesized Graph 2:")
for src in synthesized_graph2.graph:
    for tgt in synthesized_graph2.graph[src]:
        print(f"{src} -> {tgt}")
