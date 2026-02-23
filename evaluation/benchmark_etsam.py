import subprocess
import pandas as pd
import os
import time

from benchmark_utils import CPUMonitor, GPUMonitor, find_first_mrc_file

NVIDIA_GPU_ID = 0

def run_etsam_benchmark():
    """Run etsam.py benchmark on all entries in gt_testset.csv"""
    
    # Paths
    csv_path = "data/testset.csv"
    base_data_dir = "data/collection"
    base_output_dir = "results/etsam_benchmark"
    benchmark_output_csv = "results/etsam_benchmark/benchmark_results.csv"
    
    # Initialize GPU monitor
    gpu_monitor = GPUMonitor(device_id=NVIDIA_GPU_ID)
    cpu_monitor = CPUMonitor(interval=0.1, include_children=True)
    
    # Read CSV
    df = pd.read_csv(csv_path)
    
    print(f"Found {len(df)} entries in test dataset")
    print("=" * 80)
    
    # Store benchmark results
    benchmark_results = []
    
    # Process each entry
    for idx, row in df.iterrows():
        dataset_id = row['dataset_id']
        run_id = row['run_id']
        entry_num = idx + 1
        
        print(f"==> [{entry_num}/{len(df)}] Processing dataset_id={dataset_id}, run_id={run_id}")
        
        # Construct input tomogram path
        tomogram_dir = os.path.join(base_data_dir, str(dataset_id), str(run_id), "tomogram")
        input_file = find_first_mrc_file(tomogram_dir)
        
        if input_file is None:
            print(f"--> ERROR: No .mrc file found in {tomogram_dir}")
            benchmark_results.append({
                'dataset_id': dataset_id,
                'run_id': run_id,
                'time': None,
                'gpu_memory': None,
                'cpu_memory': None,
            })
            continue
        
        if not os.path.exists(input_file):
            print(f"--> ERROR: Input file does not exist: {input_file}")
            benchmark_results.append({
                'dataset_id': dataset_id,
                'run_id': run_id,
                'time': None,
                'gpu_memory': None,
                'cpu_memory': None,
            })
            continue
        
        print(f"--> Input file: {input_file}")
        
        # Construct output directory (organized by dataset_id/run_id)
        output_dir = os.path.join(base_output_dir, str(dataset_id), str(run_id))
        os.makedirs(output_dir, exist_ok=True)
        print(f"--> Output directory: {output_dir}")
        
        # Run etsam.py with timing and GPU monitoring
        cmd = [
            "python", "etsam.py",
            input_file,
            "--output-dir", output_dir,
        ]
        
        print(f"--> Running: {' '.join(cmd)}")
        
        # Measure execution time
        start_time = time.perf_counter()
        
        try:
            # Use Popen to get the process PID
            process = subprocess.Popen(
                cmd,
                stdout=None,
                stderr=None,
                text=True
            )
            
            gpu_monitor.start(process.pid)
            cpu_monitor.start(process.pid)
            
            return_code = process.wait()
            
            end_time = time.perf_counter()
            elapsed_time = end_time - start_time
            max_gpu_memory = gpu_monitor.stop()
            max_cpu_memory = cpu_monitor.stop()
            
            if return_code != 0:
                raise subprocess.CalledProcessError(return_code, cmd)
            
            print(f"--> Successfully completed")
            print(f"--> Time: {elapsed_time:.2f} seconds")
            print(f"--> Max GPU Memory: {max_gpu_memory:.2f} MB")
            print(f"--> Max CPU Memory: {max_cpu_memory:.2f} MB")
            
            benchmark_results.append({
                'dataset_id': dataset_id,
                'run_id': run_id,
                'time': elapsed_time,
                'gpu_memory': max_gpu_memory,
                'cpu_memory': max_cpu_memory,
            })

        except Exception as e:
            end_time = time.perf_counter()
            elapsed_time = end_time - start_time
            max_gpu_memory = gpu_monitor.stop()
            max_cpu_memory = cpu_monitor.stop()
            
            print(f"--> ERROR: {str(e)}")
            print(f"--> Time: {elapsed_time:.2f} seconds")
            print(f"--> Max GPU Memory: {max_gpu_memory:.2f} MB")
            print(f"--> Max CPU Memory: {max_cpu_memory:.2f} MB")
            
            benchmark_results.append({
                'dataset_id': dataset_id,
                'run_id': run_id,
                'time': elapsed_time,
                'gpu_memory': max_gpu_memory,
                'cpu_memory': max_cpu_memory,
            })
        
        print("-" * 80)
    
    # Save benchmark results to CSV
    print("\n" + "=" * 80)
    print("Saving benchmark results...")
    results_df = pd.DataFrame(benchmark_results)
    os.makedirs(os.path.dirname(benchmark_output_csv), exist_ok=True)
    results_df.to_csv(benchmark_output_csv, index=False)
    print(f"Benchmark results saved to: {benchmark_output_csv}")


if __name__ == "__main__":
    run_etsam_benchmark()
