import os
import sys
import math
import heapq
import copy
import json
import pandas as pd
from typing import List, Tuple, Dict, Set
import networkx as nx
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
                max_path_length: int, excluded_edges: Set[Tuple] = None,
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
        s_ntype=self.G.nodes[source].get('type', 'unknow')
        if target is not None:
            t_ntype=self.G.nodes[target].get('type', 'unknow') 
        path_type={node: [] for node in self.nodes}
        dist[source] = 0

        # Priority queue setup
        pq = [(0, source)]
        source_type=self.G.nodes[source].get('type', 'unknow')
        num=0
        '''
        print('*'*60)
        print(max_path_length)
        print([source,target])
        print(node_seq)
        print(excluded_edges)
        print(excluded_nodes)
        print(prev['disease:MESH:D002583,OMIM:603956'])
        '''
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
            if cur_match_len>=max_path_length:
                continue
            if target is None and current != source:
                cur_diff_len=abs(max_path_length-cur_match_len)
                if cur_diff_len<best_len:
                    best_len=cur_diff_len
                    best_dist = current_dist
                    best_node = current
                elif cur_diff_len==best_len:
                    if current_dist < best_dist:
                        best_dist = current_dist
                        best_node = current
            #print([prev[current],current_dist,current],path_type[current])
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
                
                '''
                if neighbor=='disease:OMIM:603956,MESH:D002583':
                    print(['*',current_dist,current,prev[current]])
                '''
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
        #print(dist[target],path)
        return path, dist[target]

    def yen_ksp(self, source: str, target: str,  node_seq: List, K: int = 10, 
                max_path_length: int = 4) -> List[Tuple[List[str], float]]:
        """
        Yen's K Shortest Paths Algorithm - Main Function

        Parameters:
            source: starting node
            target: destination node
            K: number of shortest paths to find
            max_path_length: maximum allowed path length (to limit search depth)
        
        Returns:
            A list of tuples, each containing:
            - path: list of nodes from source to target
            - total_weight: cumulative weight/cost of the path
        """
        # Check if the node exists
        #if source not in self.G or target not in self.G:
        if source not in self.G:
            print(f"Error: node '{source}' or '{target}' does not exist")
            return []
        
        # 1. Find the first shortest path
        path_0, weight_0 = self.dijkstra(source, target, node_seq, max_path_length)
        #print(path_0)
        if not path_0:
            print(f"No path found from {source} to {target}")
            return []
        
        # Limit Path Length
        if len(path_0) - 1 > max_path_length:
            print(f"First path length {len(path_0)-1} exceeds limit {max_path_length}")
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
                        spur_node, target, node_seq[i:], max_path_length-i, excluded_edges, excluded_nodes
                    )
                else:
                    #print(spur_node,target,node_seq,max_path_length-i,excluded_edges,excluded_nodes)
                    spur_path, spur_weight = self.dijkstra(
                        spur_node, target, node_seq, max_path_length-i, excluded_edges, excluded_nodes
                    )
                #print('*',[spur_node,target])
                #print([spur_weight,spur_path])
                # 3e. If a valid path exists
                if spur_path and spur_weight < float('inf'):
                    # Combine the root path and divergence path (remove the duplicated divergence node)
                    total_path = root_path[:-1] + spur_path
                    total_weight = self._calculate_path_weight(total_path)
                    # Check the path length constraint
                    if len(total_path) - 1 <= max_path_length:
                        # Add to the candidate list
                        if (total_weight, total_path) not in B:
                            heapq.heappush(B, (total_weight, total_path))
            
            # 4. Select the shortest path from the candidate list
            if not B:
                break  
            
            # Choose the lowest-weight candidate path
            '''
            best_weight, best_path = heapq.heappop(B)
            
            # 避免重复
            duplicate = False
            for weight, path in A:
                if path == best_path:
                    duplicate = True
                    break
            
            if not duplicate:
                A.append((best_weight, best_path))
            '''
            for weightB, pathB in B:
                not_exist=True
                for weightA, pathA in A:
                    if pathB == pathA:
                        not_exist=False
                        break
                if not_exist:
                    A.append((weightB, pathB))
        return A
    
    def _calculate_path_weight(self, path: List[str]) -> float:
        """Calculate the total weight of the path"""
        total = 0.0
        for i in range(len(path) - 1):
            u, v = path[i], path[i+1]
            weight = self.G.get_edge_data(u, v, {}).get(self.weight_key, 1.0)
            total += 1/weight
        return total
    
    def get_path_info(self, path: List[str]) -> Dict:
        """To obtain detailed information about a path."""
        info = {
            'nodes': path,
            'length': len(path) - 1,  # edge length
            'node_types': [],
            'edge_weights': [],
            'name':[],
        }
        
        # node type
        for node in path:
            node_type = self.G.nodes[node].get('type', 'unknow')
            name=self.G.nodes[node].get('name', 'unknow')
            info['node_types'].append(node_type)
            info['name'].append(name)
        
        # edge weights
        for i in range(len(path) - 1):
            u, v = path[i], path[i+1]
            weight = self.G.get_edge_data(u, v, {}).get(self.weight_key, 1.0)
            info['edge_weights'].append(weight)
        
        return info

class BiomedicalPathFinder(YensKSP):
        
    
    def find_topk_paths(self, source: str, target: str, node_seq: List, K: int = 10,
                       max_hop: int = 3, min_hop: int = 1) -> List[Dict]:
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
        #if target:
        print(f"Searching for the top-{K} paths from '{source}' to '{target}'...")
        print(f"Length constraint: {min_hop} to {max_hop} hops")
        print("-" * 60)
        
        # Find the initial path using Yen's algorithm
        raw_paths = self.yen_ksp(source, target, node_seq, max(K*3,30), max_hop)  # Find more for filtering
        if not raw_paths:
            return []

            
        detailed_paths = []
        for weight, path in raw_paths:
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
            # Calculate semantic score
            semantic_score = self._calculate_semantic_score(path_info)
            path_info['semantic_score'] = semantic_score
            #path_info['weight'] = weight
            
            import numpy as np
            path_weights=[w if w<=80 else 80 for w in path_info['edge_weights']]
            path_score = np.prod(path_weights) ** (1.0 / len(path_weights))
            #path_score = path_score - 2 *len(path_weights)
            path_info['combined_score']=path_score+semantic_score

            detailed_paths.append(path_info)
        
        #Sort by combined_score
        #detailed_paths.sort(key=lambda x: [x['start_edge_weight'],x['end_edge_weight']], reverse=True)
        detailed_paths.sort(key=lambda x: x['combined_score'], reverse=True)
        topk_paths = detailed_paths[:K]
        
        return topk_paths
                

    def _calculate_semantic_score(self, path_info: Dict) -> float:
        """Calculate semantic score"""
        type_score = 0
        node_types = path_info['node_types']
        
        # 1. The importance of node types.
        core_types = {'gene', 'chemical', 'disease', 'mutation', 'gene ontology','celltype'}
        core_count = sum(1 for t in set(node_types) if t in core_types)
        type_score += 2.5 * core_count
        
        pattern_bonus=0
        '''
        gene_pattern=True
        for i in range(path_info['length']):
            if node_types[i]=='disease' and node_types[i+1] in ['gene','chemical']:
                if node_types[i+1]=='gene':
                    pattern_bonus += 5
                else:
                    pattern_bonus += 4
            if node_types[i]=='gene':
                if node_types[i+1] in ['gene','chemical','gene ontology','disease']:
                    if node_types[i+1]=='gene':
                        if gene_pattern:
                            pattern_bonus += 5
                            gene_pattern=False
                    else:
                        pattern_bonus += 5
                elif node_types[i+1] in ['mutation','celltype']:
                    pattern_bonus += 4
            if node_types[i]=='chemical' and node_types[i+1] in ['gene','disease']:
                pattern_bonus += 4
            if node_types[i]=='mutation' and node_types[i+1] in ['gene','disease']:
                pattern_bonus += 4
        '''
        score = type_score + pattern_bonus
        
        return score
    
    
    def display_paths(self, paths: List[Dict], show_details: bool = True):
        print(f"\n{'='*80}")
        print(f"Path Search Results (Showing {len(paths)} items)")
        print(f"{'='*80}\n")
        
        for idx, path_info in enumerate(paths, 1):
            print(f"path #{idx}:")
            print(f"  Combined score: {path_info['combined_score']:.4f}")
            print(f"  Path length: {path_info['length']} hop")
            
           
            print(f"  path nodes: ", end="")
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

        
        return color_map.get(node_type, '0')  # 默认无颜色
    
    def save_paths_to_file(self, case, paths: List[Dict], filename: str):
        """将路径保存到文件"""
        import json
        import csv
        
        all_nodes=[]
        all_edges=[]
        all_edge_weights=[]
        for path in paths:
            path_nodes=path['nodes']
            path_node_type=path['node_types']
            path_node_name=path['name']
            weights=path['edge_weights']
            all_nodes+=[(path_nodes[i],path_node_type[i],path_node_name[i])for i in range(path['length']+1)]
            for i in range(path['length']):
                edge=(path_nodes[i],path_nodes[i+1])
                edge=sorted(edge)
                weight=weights[i]
                if edge not in all_edges:
                    all_edges.append(edge)
                    all_edge_weights.append(list(edge)+[min(weight,80)])
        all_nodes=set(all_nodes)
        all_nodes=pd.DataFrame(all_nodes,columns=['ID','Lable','name'])
        all_edge_weights=pd.DataFrame(all_edge_weights,columns=['source','target','weight'])
        #all_nodes.to_csv(f'../case/{case}/pathfinder/node.csv',index=False)
        #all_edge_weights.to_csv(f'../case/{case}/pathfinder/edge.csv',index=False)
        


def load_graph(case):
    edge_file=f'../case/{case}/pathfinder/graph.edgelist'
    node_file=f'../case/{case}/pathfinder/graph.nodelist'
    G = nx.Graph() 
    G = nx.read_edgelist(edge_file, data=[("weight", float)])
    G = G.to_undirected()
    for node in G:
        node_type=node.split(':')[0]
        G.nodes[node]['type'] = node_type
        if node_type in ['gene','species','mutation']:
            G.nodes[node]['name']=''
    with open(node_file,encoding='utf-8') as fp:
        for line in fp:
            line=line.strip().split('\t')
            if 'gene' in line[0] or 'species' in line[0]:
                continue
            if line[0] in G:
                if 'mutation' in line[0]:
                    G.nodes[line[0]]['name']=line[0].replace('mutation:','')
                else:
                    if len(line)<2:
                        print(line)
                    G.nodes[line[0]]['name']=line[1]
    return G

    


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--case',help='Specify case name',required=True)
    parser.add_argument('-s', '--source', type=str, required=True, help='Source node')
    parser.add_argument('-t', '--target', type=str, help='Target node')
    parser.add_argument('-n', '--node_types', type=str, help='Node type sequence, e.g.: -n disease+gene+chemical')
    parser.add_argument('-i', '--input', type=str, help='Input file path, examples: D002583||C401859||disease+gene+chemical')
    
    parser.add_argument('-k', '--path_num', type=int, default=10, help='The number of paths to be found (default: 10)')
    parser.add_argument('-max', '--max_hop', type=int, default=2, help='Maximum path hop limit')
    parser.add_argument('-min', '--min_hop', type=int, default=1, help='Minimum path hop limit')
    parser.add_argument('-d', '--display', action='store_true', default=False, help='Display the results in console')
    args = parser.parse_args()

    if not args.source and not args.input:
        parser.error("Either -s/--source or -i/--input must be provided.")
    
    queries=[]
    if args.input:
        input_file=args.input
        with open(input_file, 'r') as f:
            for line in f:
                if line.strip()=='':
                    continue
                source, target, node_types = line.strip('\n').split('|')
                if source not in self.G:
                    print(f"Error: node '{source}' does not exist")
                    if target!='' and target not in self.G:
                        print(f"Error: node '{target}' does not exist")
                    sys.exit(1)
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
    if len(queries)==0:
        print('No data was provided.')
    G=load_graph(args.case)
    finder = BiomedicalPathFinder(G)

    paths_file=f'../case/{args.case}/pathfinder/found_paths_.json'
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
        with open(paths_file, 'w') as file:
            json.dump(results, file)
        finder.save_paths_to_file(args.case, paths, paths_file)
        print(f'Path search results have been written to: {paths_file}')
    else:
        print("Path not found, file not written")


# 主程序
if __name__ == "__main__":
    #finder = run_yens_algorithm_example()
    main()
