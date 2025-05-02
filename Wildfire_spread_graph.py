import sys
import os

print("--- Runtime Environment Check ---")
print(f"Python Executable: {sys.executable}")
print("System Path (sys.path):")
for path in sys.path:
    print(f"  - {path}")
print("--- End Runtime Environment Check ---\n")

# draw the spread graph of wildfire

import argparse

from data_use.data_path import load_data_path
from modules.data_load import load_and_enrich_data
from utiles.estimate_time import estimate_fire_spread_times
from modules.graph_module import graph_module


def main():
    # 0. argparser
    # Create an instance that can receive argument values
    parser = argparse.ArgumentParser(description='Argparser')

    # Set the argument values to be input (default value can be set)
    parser.add_argument('--data_number', type=int, default=1)
    parser.add_argument('--train_or_test', type=str, default="train")
    parser.add_argument('--draw_figure', type=str, default="N")
    parser.add_argument('--save_figure', type=str, default="N")

    # Save the above in args
    args = parser.parse_args()

    # Output the value of the input arguments
    data_number = args.data_number
    train_or_test = args.train_or_test
    draw_figure = args.draw_figure
    save_figure = args.save_figure


    # 1. Collecting data
    csv_path = load_data_path(data_number)
    df = load_and_enrich_data(
        csv_path=csv_path,
        date_col='acq_date',
        time_col='acq_time',
        lat_col='latitude',
        lon_col='longitude'
    )

    if df is None:
        print("[ERROR] Data loading and enrichment failed. Exiting.")
        exit()

    df = df[df['confidence'].isin(['h', 'n'])]
    df = estimate_fire_spread_times(df)
    

    # 2. Forming a graph
    graph_module(df, data_number)
    


if __name__ == '__main__':
    main()