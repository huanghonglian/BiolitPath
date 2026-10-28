import heapq
from datetime import datetime
import json
import pickle
from pathlib import Path
from typing import List, Tuple, Dict, Set
import networkx as nx
import numpy as np
import argparse

class YensKSP:
    """Yen's K Shortest Paths Algorithm Implementation"""

    def __init__(self, G: nx.Graph, weight_key: str = 'weight'):
        """
        Initialize the algorithm

        Parameter:
            G: NetworkX graph
            weight_key: The key name of the edge weight attribute
        """
        self.G = G
        self.weight_key = weight_key
        self.nodes = list(G.nodes())

    def dijkstra(self, source: str, target: str, node_seq: List,
                max_hop: int, excluded_edges: Set[Tuple] = None,
                excluded_nodes: Set[str] = None) -> Tuple[List[str], float]:
        """
        Dijkstra's algorithm with support for excluding specific edges and nodes

        Parameters:
            source: starting node
            target: target/destination node
            excluded_edges: set of edges to exclude
            excluded_nodes: set of nodes to exclude

        Returns:
            (list of nodes in the path, total path weight/cost)
        """
        if excluded_edges is None:
            excluded_edges = set()
        if excluded_nodes is None:
            excluded_nodes = set()
        # Initialize distances & predecessors
        dist = {node: float('inf') for node in self.nodes}
        prev = {node: None for node in self.nodes}
        if target is not None:
            t_ntype=self.G.nodes[target].get('type', 'unknow')
        path_type={node: [] for node in self.nodes}
        dist[source] = 0

        # Priority queue setup
        pq = [(0, source)]
        best_node=None
        best_dist=float('inf')
        best_len=10
        while pq:
            current_dist, current = heapq.heappop(pq)
            cur_match_len=len(path_type[current])
            #If the current node being processed is the target
            if current == target and target is not None:
                break
            # If the distance extracted from the queue is worse than the best known distance
            if current_dist > dist[current]:
                continue
            if target is None and cur_match_len>=max_hop+1:
                continue
            if target is not None and cur_match_len>=max_hop:
                continue
            if target is None and current != source:
                cur_diff_len=abs(max_hop+1-cur_match_len)
                if cur_diff_len<best_len:
                    best_len=cur_diff_len
                    best_dist = current_dist
                    best_node = current
                elif cur_diff_len==best_len:
                    if current_dist < best_dist:
                        best_dist = current_dist
                        best_node = current
            # For each neighbor of the current node:
            for neighbor in self.G.neighbors(current):
                # Skip if the neighbor is in excluded_nodes
                if neighbor in excluded_nodes:
                    continue
                # Skip if the edge (u, v) is in excluded_edges
                if (current, neighbor) in excluded_edges or (neighbor, current) in excluded_edges:
                    continue

                # Obtain edge weights
                edge_data = self.G.get_edge_data(current, neighbor)
                if not edge_data:
                    continue

                weight = edge_data.get(self.weight_key, 1.0)
                ntype=self.G.nodes[neighbor].get('type', 'unknow')

                if node_seq is not None and cur_match_len+1<len(node_seq):
                    if ntype!=node_seq[cur_match_len+1]:
                        continue
                if target is not None and neighbor!=target and ntype==t_ntype:
                    continue

                # Calculate the new distance
                new_dist = current_dist + 1/min(weight,80)

                # If a shorter path is found
                if new_dist < dist[neighbor]:
                    if ntype not in path_type[current] or ntype=='gene':
                        dist[neighbor] = new_dist
                        prev[neighbor] = current
                        heapq.heappush(pq, (new_dist, neighbor))
                        path_type[neighbor]=path_type[current][:]
                        path_type[neighbor].append(ntype)

        if target is None:
            if best_node is None or best_dist == float('inf'):
                return [], float('inf')

            path = []
            cur = best_node
            while cur is not None:
                path.append(cur)
                cur = prev[cur]
            path.reverse()
            return path, best_dist

        # If the target cannot be achieved
        if dist[target] == float('inf'):
            return [], float('inf')

        # Rebuilding Path
        path = []
        current = target
        while current is not None:
            path.append(current)
            current = prev[current]
        path.reverse()
        return path, dist[target]

    def yen_ksp(self, source: str, target: str,  node_seq: List, K: int = 10,
                max_hop: int = 4) -> List[Tuple[List[str], float]]:
        """
        Yen's K Shortest Paths Algorithm - Main Function

        Parameters:
            source: starting node
            target: destination node
            K: number of shortest paths to find
            max_hop: maximum allowed path hop (to limit search depth)

        Returns:
            A list of tuples, each containing:
            - path: list of nodes from source to target
            - total_weight: cumulative weight/cost of the path
        """
        # Check if the node exists
        if source not in self.G:
            print(f"Error: node '{source}' or '{target}' does not exist")
            return []

        # 1. Find the first shortest path
        path_0, weight_0 = self.dijkstra(source, target, node_seq, max_hop)
        if not path_0:
            print(f"No path found from {source} to {target}")
            return []

        # Limit Path Length
        if len(path_0) - 1 > max_hop:
            print(f"First path length {len(path_0)-1} exceeds limit {max_hop}")
            return []

        A = [(weight_0, path_0)]  # Main list to store the found paths
        B = []  # Candidate paths list
        # 2. Find the remaining K-1 paths
        for k in range(1, K):
            if len(A) < k:
                break  # No more paths available

            prev_path = A[k-1][1]  # The (k-1)-th shortest path
            # 3. Generate candidate paths for each deviation point in prev_path
            for i in range(len(prev_path) - 1):
                # 3a. Root path
                spur_node = prev_path[i]
                root_path = prev_path[:i+1]

                # 3b. Exclude previously used edges
                excluded_edges = set()
                for _, path in A:
                    if len(path) > i and path[:i+1] == root_path:
                        u, v = path[i], path[i+1] if i+1 < len(path) else None
                        if u and v:
                            excluded_edges.add((u, v))
                            excluded_edges.add((v, u))  # Undirected graph

                # 3c. Exclude previously used nodes (except the deviation point)
                excluded_nodes = set(root_path[:-1])

                # 3d. Compute the shortest path from the divergence node to the target node
                if node_seq:
                    spur_path, spur_weight = self.dijkstra(
                        spur_node, target, node_seq[i:], max_hop-i, excluded_edges, excluded_nodes
                    )
                else:
                    spur_path, spur_weight = self.dijkstra(
                        spur_node, target, node_seq, max_hop-i, excluded_edges, excluded_nodes
                    )
                # 3e. If a valid path exists
                if spur_path and spur_weight < float('inf'):
                    # Combine the root path and divergence path (remove the duplicated divergence node)
                    total_path = root_path[:-1] + spur_path
                    total_weight = self._calculate_path_weight(total_path)
                    # Check the path length constraint
                    if len(total_path) - 1 <= max_hop:
                        # Add to the candidate list
                        if (total_weight, total_path) not in B:
                            heapq.heappush(B, (total_weight, total_path))

            # 4. Select the shortest path from the candidate list
            if not B:
                break

            # Choose exactly one globally lowest-cost candidate per iteration.
            # Discard a stale duplicate if it has already been accepted into A.
            accepted_paths = {tuple(path) for _, path in A}
            while B:
                best_weight, best_path = heapq.heappop(B)
                if tuple(best_path) not in accepted_paths:
                    A.append((best_weight, best_path))
                    break
            else:
                break
        return A

    def _calculate_path_weight(self, path: List[str]) -> float:
        """Calculate the total weight of the path"""
        total = 0.0
        for i in range(len(path) - 1):
            u, v = path[i], path[i+1]
            weight = self.G.get_edge_data(u, v, {}).get(self.weight_key, 1.0)
            total += 1 / min(weight, 80)
        return total

    def get_path_info(self, path: List[str]) -> Dict:
        """To obtain detailed information about a path."""
        info = {
            'nodes': path,
            'length': len(path) - 1,  # edge length
            'node_types': [],
            'edge_weights': [],
            'name':[],
            'node_attributes': [],
        }

        # node type
        for node in path:
            node_type = self.G.nodes[node].get('type', 'unknow')
            name=self.G.nodes[node].get('name', 'unknow')
            info['node_types'].append(node_type)
            info['name'].append(name)
            info['node_attributes'].append(dict(self.G.nodes[node]))

        # edge weights
        for i in range(len(path) - 1):
            u, v = path[i], path[i+1]
            weight = self.G.get_edge_data(u, v, {}).get(self.weight_key, 1.0)
            info['edge_weights'].append(weight)

        return info

class BiomedicalPathFinder(YensKSP):

    def find_topk_paths(self, source: str, target: str, node_seq: List, K: int = 10,
                       max_hop: int = 2, min_hop: int = 1) -> List[Dict]:
        """
        Search for the Top-K paths and return the detailed information

        Parameters:
            source: start
            target: end
            K: the number of pathway
            max_hop: max k-hop
            min_hop: min k-hop

        Return:
            List of path details
        """
        print(f"Searching for the top-{K} paths from '{source}' to '{target}'...")
        print(f"Length constraint: {min_hop} to {max_hop} hops")
        print("-" * 60)

        # Find the initial path using Yen's algorithm
        raw_paths = self.yen_ksp(source, target, node_seq, max(K+30,30), max_hop)  # Find more for filtering
        if not raw_paths:
            return []

        detailed_paths = []
        for _, path in raw_paths:
            # Skip the too short paths
            if len(path) - 1 < min_hop:
                continue
            node_types=[]
            if_continue=False
            for node in path:
                node_type = self.G.nodes[node].get('type', 'unknow')
                if node_type in node_types and node_type!='gene':
                    if_continue=True
                node_types.append(node_type)
            if if_continue:
                continue
            path_info = self.get_path_info(path)
            path_weights=[w if w<=80 else 80 for w in path_info['edge_weights']]
            path_score = np.prod(path_weights) ** (1.0 / len(path_weights))
            path_info['weight'] = path_score

            detailed_paths.append(path_info)

        detailed_paths.sort(key=lambda x: x['weight'], reverse=True)
        topk_paths = detailed_paths[:K]

        return topk_paths

    def display_paths(self, paths: List[Dict], show_details: bool = True):
        print(f"\n{'='*80}")
        print(f"Path Search Results (Showing {len(paths)} items)")
        print(f"{'='*80}\n")

        for idx, path_info in enumerate(paths, 1):
            print(f"path #{idx}:")
            print(f"  Path score: {path_info['weight']:.4f}")
            print(f"  Path length: {path_info['length']} hop")

            print("  path nodes: ", end="")
            nodes = path_info['nodes']
            node_types = path_info['node_types']
            node_names=path_info['name']
            for i, (node, ntype) in enumerate(zip(nodes, node_types)):
                color_code = self._get_color_code(ntype)
                print(f"\033[{color_code}m{node}\033[0m", end="")
                if i < len(nodes) - 1:
                    print(" → ", end="")
            print()
            for i, name in enumerate(node_names):
                print(name, end="")
                if i < len(node_names) - 1:
                    print(" → ", end="")
            print()

            if show_details and len(path_info['edge_weights']) > 0:
                print(f"  Edge weight: {path_info['edge_weights']}")
            mutation_genes = [
                attributes.get('corresponding_gene')
                for attributes in path_info.get('node_attributes', [])
                if attributes.get('type') == 'mutation'
                and attributes.get('corresponding_gene')
            ]
            if mutation_genes:
                print(f"  Mutation corresponding gene: {mutation_genes}")

            print(f"{'-'*60}")

    def _get_color_code(self, node_type: str) -> str:
        color_map = {
            'disease': '91',
            'gene': '92',
            'mutation': '93',
            'chemical': '94',
            'cell_type': '95',
            'gene_ontology': '96',
            'body_part': '97',
            'species': '90',
            'food': '98',
            'location': '99',
            'treatment': '100',
            'diagnose': '101',
            'unknow': '102',
        }

        return color_map.get(node_type, '0')


def load_graph(case):
    """Load the cached graph, or build it from text files and cache it."""
    pickle_file = Path(f'../case/{case}/pathfinder/graph.pkl')
    if pickle_file.is_file():
        with pickle_file.open('rb') as fp:
            return pickle.load(fp)

    edge_file=f'../case/{case}/pathfinder/graph.edgelist'
    node_file=f'../case/{case}/pathfinder/graph.nodelist'
    G = nx.read_edgelist(edge_file, data=[("weight", float)])
    for node in G:
        node_type=node.split(':')[0]
        G.nodes[node]['type'] = node_type
        G.nodes[node]['name']=''
    with open(node_file,encoding='utf-8') as fp:
        for line in fp:
            line=line.rstrip('\n').split('\t')
            node_id = line[0]
            node_type = node_id.split(':', 1)[0]
            if node_type in ['gene', 'species']:
                continue
            if node_id in G:
                if node_type == 'mutation':
                    mutation_id = (
                        line[2] if len(line) > 2 and line[2]
                        else node_id.replace('mutation:', '', 1)
                    )
                    G.nodes[node_id]['name'] = (
                        line[1] if len(line) > 1 and line[1]
                        else mutation_id
                    )
                    G.nodes[node_id]['mutation_id'] = mutation_id
                    G.nodes[node_id]['corresponding_gene'] = (
                        line[3] if len(line) > 3 else ''
                    )
                    G.nodes[node_id]['corresponding_genes'] = [
                        gene_id for gene_id in
                        G.nodes[node_id]['corresponding_gene'].split(';')
                        if gene_id
                    ]
                    G.nodes[node_id]['subtype'] = (
                        line[4] if len(line) > 4 else ''
                    )
                else:
                    if len(line)<2:
                        print(line)
                    else:
                        G.nodes[node_id]['name']=line[1]
    with pickle_file.open('wb') as fp:
        pickle.dump(G, fp, protocol=pickle.HIGHEST_PROTOCOL)
    return G

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--case',help='Specify case name',required=True)
    parser.add_argument('-s', '--source', type=str, help='Source node')
    parser.add_argument('-t', '--target', type=str, help='Target node')
    parser.add_argument('-n', '--node_types', type=str, help='Node type sequence, e.g.: -n disease+gene+chemical')
    parser.add_argument('-i', '--input', type=str, help='Query file with one source|target|node_types record per line')

    parser.add_argument('-k', '--path_num', type=int, default=10, help='The number of paths to be found (default: 10)')
    parser.add_argument('-max', '--max_hop', type=int, default=2, help='Maximum path hop limit')
    parser.add_argument('-min', '--min_hop', type=int, default=1, help='Minimum path hop limit')
    parser.add_argument('-d', '--display', action='store_true', default=False, help='Display the results in console')
    parser.add_argument('-o', '--output', type=Path, help='Output JSON path, relative to the current working directory')
    args = parser.parse_args()

    if not args.source and not args.input:
        parser.error("Either -s/--source or -i/--input must be provided.")

    G = load_graph(args.case)
    queries=[]
    if args.input:
        input_file=args.input
        with open(input_file, 'r') as f:
            for line in f:
                if line.strip()=='':
                    continue
                source, target, node_types = line.strip('\n').split('|')
                if source not in G:
                    parser.error(f"Source node '{source}' does not exist")
                elif target and target not in G:
                    parser.error(f"Target node '{target}' does not exist")
                else:
                    if node_types=='':
                        node_types=None
                    else:
                        node_types=node_types.split('+')
                    if target=='':
                        target=None
                    queries.append([source, target, node_types])

    else:
        source = args.source
        target = args.target
        node_types = args.node_types
        if node_types:
            node_types=node_types.strip().split('+')
            node_types=[nt.strip() for nt in node_types]
        queries.append([source, target, node_types])
    if not queries:
        parser.error('No data was provided.')
    finder = BiomedicalPathFinder(G)

    paths_file = args.output
    if paths_file is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        paths_file = Path(f'../case/{args.case}/pathfinder/found_paths_top{args.path_num}_{timestamp}.json')
    results=[]
    for source, target, node_types in queries:
        print(f"source: {source} → target: {target}, node_types={node_types} ")
        paths = finder.find_topk_paths(
                source=source,
                target=target,
                node_seq=node_types,
                K=args.path_num,
                max_hop=args.max_hop,
                min_hop=args.min_hop,
            )
        if not paths:
            print("No path found")
        else:
            if args.display:
                finder.display_paths(paths, show_details=True)
            result={'source':source,'target':target,'node_types':node_types,'top_k_paths':paths}
            results.append(result)
    if len(results)>0:

        paths_file.parent.mkdir(parents=True, exist_ok=True)
        with paths_file.open('w', encoding='utf-8') as file:
            json.dump(results, file)

        print(f'Path search results have been written to: {paths_file}')
    else:
        print("Path not found, file not written")

if __name__ == "__main__":
    main()
