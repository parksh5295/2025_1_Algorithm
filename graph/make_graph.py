import networkx as nx
import numpy as np
import random
from geopy.distance import geodesic


def build_fire_graph(nodes_df, max_distance_km=10):
    G = nx.DiGraph()
    for i, node_i_series in nodes_df.iterrows():
        G.add_node(i, **node_i_series.to_dict())
    
    for i in G.nodes():
        node_i_attrs = G.nodes[i]
        for j in G.nodes():
            if i == j: continue
            node_j_attrs = G.nodes[j]
            
            if node_j_attrs['start_time'] <= node_i_attrs['start_time']:
                continue
            
            dist = geodesic((node_i_attrs['center_latitude'], node_i_attrs['center_longitude']),
                            (node_j_attrs['center_latitude'], node_j_attrs['center_longitude'])).km
            
            if dist <= max_distance_km:
                elev_i = node_i_attrs.get('avg_elevation', 0)
                elev_j = node_j_attrs.get('avg_elevation', 0)
                elev_diff = elev_j - elev_i
                weight = dist + max(0, -elev_diff) * 0.1
                G.add_edge(i, j, weight=weight)
    
    # === Additional logic: Mandatory connection for unconnected nodes ===
    # sorted nodes
    sorted_nodes = sorted(G.nodes(), key=lambda x: G.nodes[x]['start_time'])
    
    for i, current_node in enumerate(sorted_nodes):
        if i == 0:  # first node has no previous node
            continue
            
        current_attrs = G.nodes[current_node]
        previous_nodes = sorted_nodes[:i]  # 현재 노드보다 이전 시간의 모든 노드들
        
        # 모든 이전 노드와의 거리 계산
        node_distances = []
        for prev_node in previous_nodes:
            prev_attrs = G.nodes[prev_node]
            dist = geodesic((prev_attrs['center_latitude'], prev_attrs['center_longitude']),
                            (current_attrs['center_latitude'], current_attrs['center_longitude'])).km
            node_distances.append((prev_node, dist))
        
        node_distances.sort(key=lambda x: x[1])  # 거리 순 정렬
        
        # 2. 의무 연결 체크: 연결되지 않은 노드는 가장 가까운 1개에 연결
        has_incoming_edge = any(G.has_edge(prev_node, current_node) for prev_node in previous_nodes)
        
        if not has_incoming_edge and node_distances:
            nearest_node, min_dist = node_distances[0]  # 가장 가까운 1개
            elev_prev = G.nodes[nearest_node].get('avg_elevation', 0)
            elev_curr = current_attrs.get('avg_elevation', 0)
            elev_diff = elev_curr - elev_prev
            weight = min_dist + max(0, -elev_diff) * 0.1
            G.add_edge(nearest_node, current_node, weight=weight)
            print(f"[EDGE_DEBUG] Mandatory connection: node {current_node} to nearest node {nearest_node} (distance: {min_dist:.2f}km)")
        
        # 3. 추가 랜덤 연결: 2~5개 추가 연결 (기존 연결과 중복되지 않게)
        if len(node_distances) > 1:  # 최소 2개 이상의 이전 노드가 있어야 추가 연결 가능
            # 이미 연결된 노드들 찾기
            already_connected = set()
            for prev_node in previous_nodes:
                if G.has_edge(prev_node, current_node):
                    already_connected.add(prev_node)
            
            # 연결되지 않은 노드들만 후보로 선정
            available_candidates = [(node, dist) for node, dist in node_distances 
                                  if node not in already_connected]
            
            if available_candidates:
                # 2~5개 중 랜덤 선택 (사용 가능한 후보 수 고려)
                max_additional = min(5, len(available_candidates))
                num_additional = random.randint(2, max_additional) if max_additional >= 2 else 0
                
                if num_additional > 0:
                    # 가장 가까운 후보들 중에서 랜덤 선택
                    max_candidates = min(8, len(available_candidates))  # 상위 8개 중에서 선택
                    candidate_pool = available_candidates[:max_candidates]
                    selected_additional = random.sample(candidate_pool, min(num_additional, len(candidate_pool)))
                    
                    print(f"[EDGE_DEBUG] Adding {len(selected_additional)} additional random connections for node {current_node}")
                    
                    for add_node, add_dist in selected_additional:
                        elev_prev = G.nodes[add_node].get('avg_elevation', 0)
                        elev_curr = current_attrs.get('avg_elevation', 0)
                        elev_diff = elev_curr - elev_prev
                        weight = add_dist + max(0, -elev_diff) * 0.1
                        G.add_edge(add_node, current_node, weight=weight)
                        print(f"[EDGE_DEBUG] Additional connection: node {current_node} to node {add_node} (distance: {add_dist:.2f}km)")
    
    return G
