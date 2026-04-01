import heapq
from typing import Dict, List, Tuple, Set, Optional
import networkx as nx



def dijkstra_with_node_type_constraint(
    graph: nx.Graph,
    source: str,
    target: str,
    node_types: List[str] = ['disease', 'gene', 'chemical']
) -> Tuple[Optional[List[str]], float]:
    """
    Alternative implementation that strictly enforces node type sequence order.
    Paths must follow the exact sequence pattern (can skip types but cannot reverse order).
    
    Args:
        graph: NetworkX graph with node attribute 'type'
        source: Source node
        target: Target node
        node_types: Required order of node types (must follow this sequence)
    
    Returns:
        Tuple of (path, total_weight) or (None, inf) if no valid path exists
    """
    
    def get_type_index(node: str) -> int:
        """Get index of node's type in the sequence."""
        node_type = graph.nodes[node].get('type', 'unknown')
        try:
            return node_types.index(node_type)
        except ValueError:
            return -1  # Unknown type not allowed
    
    # Priority queue: (total_weight, current_type_index, current_node, path)
    pq = [(0, 0, source, [source])]
    
    # Track visited: (node, type_index) -> best_weight
    visited = {}
    
    while pq:
        total_weight, type_idx, current_node, path = heapq.heappop(pq)
        
        # Check if valid path to current node with this type index
        state = (current_node, type_idx)
        if state in visited and visited[state] <= total_weight:
            continue
        
        visited[state] = total_weight
        
        # Check if we reached the target
        if current_node == target:
            return path, total_weight
        
        # Explore neighbors
        for neighbor in graph.neighbors(current_node):
            if neighbor in path:  # Avoid cycles
                continue
            
            neighbor_type_idx = get_type_index(neighbor)
            
            # Skip if neighbor type is unknown
            if neighbor_type_idx == -1:
                continue
            
            # Check if neighbor type follows sequence order
            if neighbor_type_idx < type_idx:
                # Cannot go backwards in sequence
                continue
            
            # Calculate edge weight
            edge_weight = graph[current_node][neighbor].get('weight', 1)
            
            new_path = path + [neighbor]
            new_weight = total_weight + edge_weight
            
            heapq.heappush(pq, (new_weight, neighbor_type_idx, neighbor, new_path))
    
    return None, float('inf')


# Example usage and testing
def create_sample_graph():
    """Create a sample graph for testing."""
    G = nx.Graph()
    
    # Add nodes with types
    nodes = {
        'd1': 'disease',
        'd2': 'disease',
        'g1': 'gene',
        'g2': 'gene',
        'c1': 'chemical',
        'c2': 'chemical',
        'mixed': 'unknown'
    }
    
    for node, node_type in nodes.items():
        G.add_node(node, type=node_type)
    
    # Add edges with weights
    edges = [
        ('d1', 'g1', 1),
        ('d1', 'g2', 2),
        ('g1', 'c1', 1),
        ('g2', 'c1', 1),
        ('c1', 'c2', 1),
        ('c1', 'mixed', 1),
        ('mixed', 'd2', 1),
        ('d1', 'd2', 3),  # Direct disease-disease edge
    ]
    
    for u, v, w in edges:
        G.add_edge(u, v, weight=w)
    
    return G


if __name__ == "__main__":
    # Test the algorithms
    G = create_sample_graph()
    node_types = ['disease', 'gene', 'chemical']
    
    
    print("\nTest 2: Strict sequence enforcement")
    print("-" * 50)
    path, weight = dijkstra_with_node_type_constraint(G, 'd1', '', node_types)
    print(path)
    print(f"Path from d1 to c2: {' -> '.join(path)}")
    print(f"Total weight: {weight}")
    for node in path:
        print(f"  {node}: {G.nodes[node].get('type', 'unknown')}")
        