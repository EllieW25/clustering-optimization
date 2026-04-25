import json
import matplotlib.pyplot as plt
import sys
import os
import csv
#HOW TO RUN
#python compare.py
#you will be prompted to enter information

def load_single_result(path):
    path = path.strip()
    if not path:
        return None
    if not os.path.exists(path):
        print(f'Error: JSON file not found: {path}')
        return None
    with open(path, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
        except Exception as e:
            print(f'Error reading {path}: {e}')
            return None

    n = data.get('n')
    k = data.get('k')
    runtime = data.get('runtime_seconds')
    memory = data.get('memory_mb')
    average_sse = data.get('average_sse')
    linkage = data.get('linkage', 'unknown')
    algorithm = data.get('algorithm', 'unknown')
    
    if n is None or runtime is None or memory is None or average_sse is None:
        print(f'Warning: {path} is missing required fields; skipping')
        return None

    # Ask if data is synthetic
    while True:
        is_synthetic = input(f"  Is this data synthetic? (y/n): ").strip().lower()
        if is_synthetic in ['y', 'n']:
            break
        print("  Please enter 'y' or 'n'")
    
    dataset = 'synthetic' if is_synthetic == 'y' else 'mall_dataset'

    return {
        'path': path,
        'n': n,
        'k': k,
        'runtime': runtime,
        'memory': memory,
        'average_sse': average_sse,
        'linkage': linkage,
        'algorithm': algorithm,
        'dataset': dataset,
    }


def plot_results(results):
    if not results:
        print('No valid results to plot.')
        return

    results.sort(key=lambda r: r['n'])
    ns = [r['n'] for r in results]
    runtimes = [r['runtime'] for r in results]
    memories = [r['memory'] for r in results]

    # Plot 1: Runtime scaling (separate window)
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(ns, runtimes, marker='o', linestyle='-', color='blue', linewidth=2)
    ax1.set_title('Runtime vs Dataset Size (n)')
    ax1.set_xlabel('Dataset size (n)')
    ax1.set_ylabel('Runtime (seconds)')
    ax1.grid(True, linestyle='--', alpha=0.5)
    fig1.tight_layout()
    
    # Plot 2: Memory scaling (separate window)
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.plot(ns, memories, marker='s', linestyle='-', color='green', linewidth=2)
    ax2.set_title('Memory Usage vs Dataset Size (n)')
    ax2.set_xlabel('Dataset size (n)')
    ax2.set_ylabel('Memory (MB)')
    ax2.grid(True, linestyle='--', alpha=0.5)
    fig2.tight_layout()

    plt.show()


def write_csv_results(results, output_file='results.csv'):
    """Write results to CSV file with columns: dataset, n, k, avg_sse, avg_runtime, memory_mb"""
    if not results:
        print('No results to write to CSV.')
        return
    
    # Sort by dataset and then by n
    results.sort(key=lambda r: (r['dataset'], r['n']))
    
    try:
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['dataset', 'n', 'k', 'avg_sse', 'avg_runtime', 'memory_mb'])
            for r in results:
                writer.writerow([
                    r['dataset'],
                    r['n'],
                    r['k'],
                    f"{r['average_sse']:.5g}",  # Format with 5 significant figures
                    f"{r['runtime']:.8g}",  # Format with 8 significant figures
                    f"{r['memory']:.5g}"  # Format memory with 5 significant figures
                ])
        print(f"\nResults saved to {output_file}")
    except Exception as e:
        print(f"Error writing CSV file: {e}")


if __name__ == '__main__':
    results = []
    
    print('Enter JSON file paths one at a time.')
    print('Enter 0 when done to generate the graph.\n')
    
    counter = 1
    while True:
        path_input = input(f'JSON file {counter} (or 0 to finish): ').strip()
        
        if path_input == '0':
            break
        
        result = load_single_result(path_input)
        if result:
            results.append(result)
            print(f"  Loaded: n={result['n']}, k={result['k']}, avg_sse={result['average_sse']:.5g}, runtime={result['runtime']:.8g}s, dataset={result['dataset']}")
            counter += 1
    
    if not results:
        print('No valid results loaded.')
        sys.exit(1)

    print(f'\nLoaded {len(results)} result(s):')
    for r in sorted(results, key=lambda r: (r['dataset'], r['n'])):
        print(f"  {r['dataset']}: n={r['n']}, k={r['k']}, avg_sse={r['average_sse']:.5g}, runtime={r['runtime']:.8g}s")

    # Write CSV results
    write_csv_results(results)

    print('\nGenerating graph...')
    plot_results(results)
