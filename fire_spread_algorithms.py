import pandas as pd
import numpy as np
import heapq
from collections import deque
from datetime import datetime, timedelta
import math
from typing import Dict, List, Tuple, Set, Optional
import sys
import os

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from prediction.prediction_utils import calculate_bearing
    from modules.data_load import load_and_enrich_data
except ImportError:
    print("Warning: Some modules not found. Using basic implementations.")

class FireSpreadAnalyzer:
    """
    Fire Spread Analysis Class
    - BFS: Real-time fire spread simulation
    - Dijkstra: Calculation of the shortest time for fire to reach the target point
    """
    
    def __init__(self, all_nodes_df: pd.DataFrame, time_step_minutes: int = 15):
        """
        Args:
            all_nodes_df: All node information (location, environmental data)
            time_step_minutes: Time unit (minutes)
        """
        self.all_nodes_df = all_nodes_df.copy()
        self.time_step_minutes = time_step_minutes
        self.node_positions = {}  # node_id -> (lat, lon)
        self.node_features = {}   # node_id -> features dict
        
        # Node Information Indexing
        for idx, row in all_nodes_df.iterrows():
            node_id = row.get('node_id', idx)
            self.node_positions[node_id] = (row['latitude'], row['longitude'])
            self.node_features[node_id] = row.to_dict()
    
    def get_neighbors(self, node_id: int, max_distance_config: float = 0.1, excluded_nodes: set = None) -> List[int]:
        """Using the neighbor search logic of the existing system"""
        if excluded_nodes is None:
            excluded_nodes = set()
            
        try:
            # Using the neighbor finder in the existing system
            from prediction.neighbor_definition import example_neighbor_finder
            
            neighbors_df = example_neighbor_finder(
                node_id, 
                self.all_nodes_df, 
                excluded_nodes, 
                max_dist_config=max_distance_config
            )
            
            if not neighbors_df.empty:
                return neighbors_df['node_id'].tolist()
            else:
                return []
                
        except ImportError:
            # Fallback: Distance-based calculation
            if node_id not in self.node_positions:
                return []
            
            neighbors = []
            current_lat, current_lon = self.node_positions[node_id]
            
            for other_id, (other_lat, other_lon) in self.node_positions.items():
                if other_id == node_id or other_id in excluded_nodes:
                    continue
                    
                # Euclidean distance calculation (in degrees)
                distance = math.sqrt(
                    (current_lat - other_lat)**2 + (current_lon - other_lon)**2
                )
                if distance <= max_distance_config:
                    neighbors.append(other_id)
            
            return neighbors
    
    def _haversine_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate distance between two points (km)"""
        R = 6371.0  # Earth radius (km)
        
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lat2, lon1, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))
        
        return R * c
    
    def calculate_spread_weight(self, source_id: int, target_id: int) -> float:
        """Use the diffusion weight calculation logic of the existing system."""
        if source_id not in self.node_features or target_id not in self.node_features:
            return 0.0
        
        source_features = self.node_features[source_id]
        target_features = self.node_features[target_id]
        
        try:
            # Use of the destination calculator in the existing system
            from prediction.prediction_utils import calculate_spread_weight, example_destination_calculator
            
            destination_metric = example_destination_calculator(source_features, target_features)
            
            # Use of basic coefficients in existing systems
            c_coeffs = {
                'c1': 0.1,  # Destination term
                'c2': 0.05, # Wind term
                'c3': 0.1,  # Temperature difference
                'c4': 0.1,  # Humidity difference
                'c5': 0.1,  # Rainfall difference
                'c6': 0.2,  # NDVI sum
                'c7': 0.05  # Elevation sum
            }
            
            weight = calculate_spread_weight(source_features, target_features, destination_metric, c_coeffs)
            return weight
            
        except ImportError:
            # Fallback: Simple distance-based calculation
            source_lat, source_lon = self.node_positions[source_id]
            target_lat, target_lon = self.node_positions[target_id]
            distance = self._haversine_distance(source_lat, source_lon, target_lat, target_lon)
            return max(0.1, 1.0 / (1.0 + distance))
    
    def bfs_fire_spread(self, initial_fire_nodes: List[int], max_steps: int = 100, 
                       spread_threshold: float = 0.3) -> Dict[int, Dict]:
        """
        Real-time fire spread simulation using BFS
        
        Args:
            initial_fire_nodes: Initial fire nodes
            max_steps: Maximum simulation steps
            spread_threshold: Spread threshold
            
        Returns:
            Dict[step, Dict[node_id, fire_info]]
        """
        print(f"BFS fire spread simulation started (initial nodes: {len(initial_fire_nodes)} nodes)")
        
        # Initialization
        fire_history = {}  # step -> {node_id: fire_info}
        burning_nodes = set(initial_fire_nodes)  # Currently burning nodes
        spread_queue = deque()  # BFS queue
        
        # Add initial fire nodes to the queue
        current_time = datetime.now()
        for node_id in initial_fire_nodes:
            spread_queue.append((node_id, 0, current_time))  # (node_id, step, ignition_time)
        
        fire_history[0] = {
            node_id: {
                'ignition_time': current_time,
                'step': 0,
                'spread_probability': 1.0,
                'latitude': self.node_positions[node_id][0],
                'longitude': self.node_positions[node_id][1]
            }
            for node_id in initial_fire_nodes
        }
        
        # BFS spread simulation
        for step in range(1, max_steps + 1):
            if not spread_queue:
                break
                
            step_ignitions = {}
            current_step_time = current_time + timedelta(minutes=step * self.time_step_minutes)
            
            # Spread processing in the current step
            queue_size = len(spread_queue)
            for _ in range(queue_size):
                if not spread_queue:
                    break
                    
                source_node, source_step, source_time = spread_queue.popleft()
                
                # Spread attempt for neighbors (using the neighbor search logic of the existing system)
                neighbors = self.get_neighbors(source_node, max_distance_config=0.1, excluded_nodes=burning_nodes)
                
                for neighbor_id in neighbors:
                    # Spread probability calculation (using the diffusion weight calculation logic of the existing system)
                    spread_prob = self.calculate_spread_weight(source_node, neighbor_id)
                    
                    if spread_prob >= spread_threshold:
                        # Spread
                        burning_nodes.add(neighbor_id)
                        spread_queue.append((neighbor_id, step, current_step_time))
                        
                        step_ignitions[neighbor_id] = {
                            'ignition_time': current_step_time,
                            'step': step,
                            'spread_probability': spread_prob,
                            'source_node': source_node,
                            'latitude': self.node_positions[neighbor_id][0],
                            'longitude': self.node_positions[neighbor_id][1]
                        }
            
            if step_ignitions:
                fire_history[step] = step_ignitions
                print(f"  Step {step}: {len(step_ignitions)} nodes spread (total {len(burning_nodes)} nodes)")
            else:
                print(f"  Step {step}: No spread")
        
        total_burned = sum(len(step_data) for step_data in fire_history.values())
        print(f"BFS simulation completed: {len(fire_history)} steps, total {total_burned} nodes lost")
        
        return fire_history
    
    '''
    def priority_queue_fire_spread(self, initial_fire_nodes: List[int], max_time_hours: float = 24.0, 
                                   spread_threshold: float = 0.3) -> Dict[float, Dict]:
        """
        Exact fire spread simulation using priority queue
        Calculate the actual spread time for each node to ignite at the exact time
        
        Args:
            initial_fire_nodes: Initial fire nodes
            max_time_hours: Maximum simulation time (hours)
            spread_threshold: Spread threshold
            
        Returns:
            Dict[time_minutes, Dict[node_id, fire_info]]
        """
        print(f"Priority queue fire spread simulation started (initial nodes: {len(initial_fire_nodes)} nodes)")
        
        # Initialization
        fire_history = {}  # time_minutes -> {node_id: fire_info}
        burning_nodes = set(initial_fire_nodes)  # Currently burning nodes
        priority_queue = []  # (ignition_time_minutes, node_id, source_node_id)
        
        # Setting initial fire nodes
        start_time = 0.0
        for node_id in initial_fire_nodes:
            heapq.heappush(priority_queue, (start_time, node_id, None))
        
        fire_history[start_time] = {
            node_id: {
                'ignition_time_minutes': start_time,
                'spread_probability': 1.0,
                'source_node': None,
                'latitude': self.node_positions[node_id][0],
                'longitude': self.node_positions[node_id][1]
            }
            for node_id in initial_fire_nodes
        }
        
        max_time_minutes = max_time_hours * 60.0
        
        # Spread simulation based on priority queue
        while priority_queue:
            current_time, current_node, source_node = heapq.heappop(priority_queue)
            
            if current_time > max_time_minutes:
                break
            
            # Skip already processed nodes (avoid duplicates)
            if current_node in burning_nodes and current_time > start_time:
                continue
                
            # Set the current node to burning state
            if current_node not in burning_nodes:
                burning_nodes.add(current_node)
                
                # Fire record
                if current_time not in fire_history:
                    fire_history[current_time] = {}
                    
                fire_history[current_time][current_node] = {
                    'ignition_time_minutes': current_time,
                    'spread_probability': 1.0,  # Already
                    'source_node': source_node,
                    'latitude': self.node_positions[current_node][0],
                    'longitude': self.node_positions[current_node][1]
                }
            
            # Spread calculation for neighbors
            neighbors = self.get_neighbors(current_node, max_distance_config=0.1, excluded_nodes=burning_nodes)
            
            for neighbor_id in neighbors:
                if neighbor_id in burning_nodes:
                    continue
                
                # Spread weight calculation
                spread_weight = self.calculate_spread_weight(current_node, neighbor_id)
                
                if spread_weight >= spread_threshold:
                    # Spread time calculation (inverse proportional to weight)
                    spread_time = self.time_step_minutes / spread_weight
                    ignition_time = current_time + spread_time
                    
                    # Add to priority queue
                    heapq.heappush(priority_queue, (ignition_time, neighbor_id, current_node))
        
        # Output sorted results
        sorted_times = sorted(fire_history.keys())
        total_burned = sum(len(step_data) for step_data in fire_history.values())
        
        print(f"Priority queue simulation completed:")
        print(f"Total simulation time: {sorted_times[-1]:.1f} minutes ({sorted_times[-1]/60:.1f} hours)")
        print(f"Total lost nodes: {total_burned} nodes")
        print(f"Fire events: {len(sorted_times)} times")
        
        return fire_history

    '''
    
    def dijkstra_fastest_path(self, source_nodes: List[int], target_node: int) -> Tuple[float, List[int], Dict]:
        """
        Calculation of the shortest time for fire to reach the target point using Dijkstra algorithm
        
        Args:
            source_nodes: Fire starting nodes
            target_node: Target node
            
        Returns:
            (shortest_time_minutes, path_node_list, detailed_info)
        """
        print(f"Dijkstra shortest path calculation: {len(source_nodes)} starting points → target ({target_node})")
        
        if target_node not in self.node_positions:
            return float('inf'), [], {"error": "Target node not found"}
        
        # Dijkstra initialization
        distances = {node_id: float('inf') for node_id in self.node_positions.keys()}
        previous = {}
        visited = set()
        priority_queue = []
        
        # Setting start nodes
        for source_node in source_nodes:
            if source_node in self.node_positions:
                distances[source_node] = 0.0
                heapq.heappush(priority_queue, (0.0, source_node))
        
        path_details = {
            'nodes_processed': 0,
            'max_queue_size': 0,
            'spread_weights': {}
        }
        
        while priority_queue:
            path_details['max_queue_size'] = max(path_details['max_queue_size'], len(priority_queue))
            current_time, current_node = heapq.heappop(priority_queue)
            
            if current_node in visited:
                continue
            
            visited.add(current_node)
            path_details['nodes_processed'] += 1
            
            # End when the target node is reached
            if current_node == target_node:
                break
            
            # Processing
            neighbors = self.get_neighbors(current_node, max_distance_config=0.1, excluded_nodes=visited)
            
            for neighbor_id in neighbors:
                if neighbor_id in visited:
                    continue
                
                # Convert spread weight to time
                spread_weight = self.calculate_spread_weight(current_node, neighbor_id)
                
                if spread_weight <= 0:
                    continue
                
                # Calculate spread time (inverse proportional to weight)
                spread_time_minutes = self.time_step_minutes / spread_weight
                new_time = current_time + spread_time_minutes
                
                if new_time < distances[neighbor_id]:
                    distances[neighbor_id] = new_time
                    previous[neighbor_id] = current_node
                    heapq.heappush(priority_queue, (new_time, neighbor_id))
                    path_details['spread_weights'][(current_node, neighbor_id)] = spread_weight
        
        # Reconstruct path
        if target_node not in previous and distances[target_node] == float('inf'):
            print(f"Cannot reach the target node")
            return float('inf'), [], path_details
        
        path = []
        current = target_node
        while current is not None:
            path.append(current)
            current = previous.get(current)
        path.reverse()
        
        total_time = distances[target_node]
        print(f"Shortest time: {total_time:.1f} minutes ({total_time/60:.1f} hours)")
        print(f"Path: {len(path)} nodes")
        
        return total_time, path, path_details
    
    def export_bfs_results(self, fire_history: Dict, output_path: str) -> None:
        """Export BFS results to CSV"""
        rows = []
        
        for step, step_data in fire_history.items():
            for node_id, fire_info in step_data.items():
                rows.append({
                    'step': step,
                    'node_id': node_id,
                    'ignition_time': fire_info['ignition_time'],
                    'spread_probability': fire_info['spread_probability'],
                    'latitude': fire_info['latitude'],
                    'longitude': fire_info['longitude'],
                    'source_node': fire_info.get('source_node', 'initial')
                })
        
        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False)
        print(f"BFS results saved to: {output_path}")
    
    def export_dijkstra_results(self, fastest_time: float, path: List[int], 
                              details: Dict, output_path: str) -> None:
        """Export Dijkstra results to CSV"""
        if not path:
            print("No path to save")
            return
        
        rows = []
        cumulative_time = 0.0
        
        for i, node_id in enumerate(path):
            if i > 0:
                prev_node = path[i-1]
                spread_weight = details['spread_weights'].get((prev_node, node_id), 0.5)
                step_time = self.time_step_minutes / spread_weight
                cumulative_time += step_time
            
            lat, lon = self.node_positions[node_id]
            rows.append({
                'step': i,
                'node_id': node_id,
                'cumulative_time_minutes': cumulative_time,
                'cumulative_time_hours': cumulative_time / 60,
                'latitude': lat,
                'longitude': lon
            })
        
        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False)
        print(f"Dijkstra results saved to: {output_path}")


def main():
    """Main execution function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Fire spread BFS & Dijkstra analysis")
    parser.add_argument('--data_number', type=int, required=True, help="Data number")
    parser.add_argument('--mode', choices=['bfs', 'dijkstra', 'both'], default='both', 
    # parser.add_argument('--mode', choices=['bfs', 'priority_queue', 'dijkstra', 'both'], default='both', 
                       help="Execution mode")
    parser.add_argument('--initial_nodes', type=str, help="Initial fire nodes (comma separated, e.g.: 1,5,10)")
    parser.add_argument('--target_node', type=int, help="Target node (Dijkstra only)")
    parser.add_argument('--max_steps', type=int, default=50, help="BFS maximum steps")
    parser.add_argument('--spread_threshold', type=float, default=0.3, help="Spread threshold")
    parser.add_argument('--output_dir', type=str, default='fire_analysis_results', 
                       help="Result save directory")
    
    args = parser.parse_args()
    
    # Load data
    try:
        from data_use.data_path import get_data_path
        data_path, _ = get_data_path(args.data_number)
        
        try:
            all_nodes_df = load_and_enrich_data(data_path)
        except:
            # Fallback: basic CSV loading
            all_nodes_df = pd.read_csv(data_path)
            if 'node_id' not in all_nodes_df.columns:
                all_nodes_df['node_id'] = all_nodes_df.index
        
        print(f"Data loaded: {len(all_nodes_df)} nodes")
    except Exception as e:
        print(f"Data load failed: {e}")
        return
    
    # Initialize analyzer
    analyzer = FireSpreadAnalyzer(all_nodes_df)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set
    if args.initial_nodes:
        initial_nodes = [int(x.strip()) for x in args.initial_nodes.split(',')]
    else:
        # Default: first 3 nodes
        initial_nodes = list(range(min(3, len(all_nodes_df))))
    
    print(f"Initial fire nodes: {initial_nodes}")
    
    # Run BFS
    if args.mode in ['bfs', 'both']:
        print("\n" + "="*50)
        print("BFS fire spread simulation")
        print("="*50)
        
        fire_history = analyzer.bfs_fire_spread(
            initial_nodes, 
            max_steps=args.max_steps,
            spread_threshold=args.spread_threshold
        )
        
        bfs_output = os.path.join(args.output_dir, f'bfs_fire_spread_data{args.data_number}.csv')
        analyzer.export_bfs_results(fire_history, bfs_output)
    
    '''
    
    # Run priority queue (exact method)
    if args.mode in ['priority_queue', 'both']:
        print("\n" + "="*50)
        print("Priority queue fire spread simulation (exact time-based)")
        print("Recommended: Calculate the actual spread time exactly")
        print("="*50)
        
        pq_fire_history = analyzer.priority_queue_fire_spread(
            initial_nodes,
            max_time_hours=24.0,
            spread_threshold=args.spread_threshold
        )
        
        pq_output = os.path.join(args.output_dir, f'priority_queue_fire_spread_data{args.data_number}.csv')
        analyzer.export_priority_queue_results(pq_fire_history, pq_output)

    '''
    
    # Run Dijkstra
    if args.mode in ['dijkstra', 'both']:
        print("\n" + "="*50)
        print("Dijkstra shortest path calculation")
        print("="*50)
        
        target_node = args.target_node if args.target_node is not None else len(all_nodes_df) - 1
        
        fastest_time, path, details = analyzer.dijkstra_fastest_path(
            initial_nodes, 
            target_node
        )
        
        if path:
            dijkstra_output = os.path.join(args.output_dir, f'dijkstra_path_data{args.data_number}.csv')
            analyzer.export_dijkstra_results(fastest_time, path, details, dijkstra_output)
            
            print(f"\nAnalysis summary:")
            print(f"  • Processed nodes: {details['nodes_processed']} nodes")
            print(f"  • Maximum queue size: {details['max_queue_size']}")
            print(f"  • Total reach time: {fastest_time:.1f} minutes")
            print(f"  • Path length: {len(path)} nodes")
    
    print(f"\nAll analysis completed! Results saved to '{args.output_dir}' folder.")


if __name__ == "__main__":
    main()