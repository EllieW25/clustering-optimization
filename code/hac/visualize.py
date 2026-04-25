import json
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import sys
import os
#HOW TO RUN
#Example execution below
#python visualize.py ..\..\data\synthetic\blobs_100.csv 100avg.json "100 point"


def visualize_clusters(csv_path, json_path, title="HAC Clusters"):
    # Check if files exist
    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found: {csv_path}")
        return
    
    if not os.path.exists(json_path):
        print(f"Error: JSON file not found: {json_path}")
        return
    
    # Load dataset
    df = pd.read_csv(csv_path)

    # Keep only numeric columns for clustering/visualization
    numeric_cols = df.select_dtypes(include=['number']).columns
    if len(numeric_cols) == 0:
        print("Error: No numeric columns found in dataset")
        return
    
    df_numeric = df[numeric_cols]
    print(f"Using {len(numeric_cols)} numeric columns: {list(numeric_cols)}")

    # Load HAC labels from JSON
    with open(json_path, "r") as f:
        hac_output = json.load(f)

    labels = hac_output["labels"]

    # Check if labels match data size
    if len(labels) != len(df_numeric):
        print(f"Error: JSON labels ({len(labels)}) don't match data points ({len(df_numeric)})")
        return

    # Reduce to 2D using PCA
    pca = PCA(n_components=2)
    X2 = pca.fit_transform(df_numeric.values)

    # Plot
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X2[:, 0], X2[:, 1], c=labels, cmap="tab10", s=10)

    plt.title(title)
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")

    # Legend for cluster IDs
    plt.legend(*scatter.legend_elements(), title="Cluster")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python visualize.py <csv_path> [title]")
        print("The script will prompt for the JSON filename.")
        print("\nExample:")
        print("  python visualize.py ../../data/synthetic/blobs_1600.csv")
        sys.exit(1)
    
    # Prompt for JSON filename
    json_name = input("Enter JSON filename (without .json extension): ").strip()
    if not json_name.endswith('.json'):
        json_name += '.json'
    
    csv_path = sys.argv[1]
    title = sys.argv[2] if len(sys.argv) > 2 else "HAC Clusters"
    
    visualize_clusters(csv_path, json_name, title)
