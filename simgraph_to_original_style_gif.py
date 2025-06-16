import argparse
import os
import pandas as pd
from utiles.estimate_time import add_datetime_column

def main():
    parser = argparse.ArgumentParser(description="Create a networkx/matplotlib style wildfire spread GIF from simgraph data (no API enrich, 원본 스타일)")
    parser.add_argument('--data_number', type=int, required=True, help="The data number to process (e.g., 1, 2, 3)")
    args = parser.parse_args()

    # 1. simgraph 파일 경로 찾기
    data_number_str = str(args.data_number).zfill(4)
    pred_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'prediction')
    simgraph_candidates = [f for f in os.listdir(pred_dir) if f.startswith('simgraph_') and data_number_str in f]
    if not simgraph_candidates:
        print(f"[ERROR] simgraph_predicted_spread_{data_number_str}.csv 파일을 찾을 수 없습니다.")
        return
    simgraph_path = os.path.join(pred_dir, simgraph_candidates[0])
    print(f"[INFO] Loading simgraph data from {simgraph_path}")

    # 2. 데이터 불러오기
    df = pd.read_csv(simgraph_path)
    if 'date' not in df.columns:
        if 'acq_date' in df.columns and 'acq_time' in df.columns:
            df = add_datetime_column(df, 'acq_date', 'acq_time')
        else:
            print("[ERROR] No 'date' or ('acq_date'+'acq_time') columns found in simgraph data.")
            return

    # 3. 시간 그룹핑 (15분 단위)
    df['date_10min'] = pd.to_datetime(df['date']).dt.floor('15T')
    interval_groups = list(df.groupby('date_10min'))
    total_intervals = len(interval_groups)
    print(f"Found {total_intervals} unique 15-min intervals to process.")
    if total_intervals == 0:
        print("No data to process.")
        return

    # 4. 누적 프레임 생성 및 네트워크/엣지/지도 포함 스냅샷
    from graph.build_graph import cluster_and_build_graph
    from graph.snapshot import draw_graph_snapshot
    from utiles.gen_gif import generate_gif_for_dataset
    cumulative_df = pd.DataFrame(columns=df.columns)
    for i, (interval, interval_df) in enumerate(interval_groups):
        sequence_id = i + 1
        cumulative_df = pd.concat([cumulative_df, interval_df], ignore_index=True)
        _processed_df, nodes_df, G = cluster_and_build_graph(cumulative_df.copy())
        draw_graph_snapshot(G, f"similar_sub_{args.data_number}", sequence_id)
        print(f"   Snapshot saved for sequence {sequence_id}")

    # 5. GIF 생성
    print(f"Generating GIF for similar_sub_animation_{args.data_number}...")
    graph_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'graph')
    generate_gif_for_dataset(
        filenumber=f"similar_sub_{args.data_number}",
        frame_base_dir=graph_dir,
        output_gif_name=f"similar_sub_animation_{args.data_number}.gif",
        duration=0.25,
        frame_image_pattern='*.png'
    )
    print(f"--- Similar sub animation GIF created as 'similar_sub_animation_{args.data_number}.gif' ---")

if __name__ == "__main__":
    main() 