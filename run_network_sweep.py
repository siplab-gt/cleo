import time
import numpy as np

# Simple template for benchmarking full network simulation sweeps
def benchmark_network_sweep():
    # 1. Define the sweep range (e.g., light intensity in mW/mm^2 or pulse duration in ms)
    stimulus_intensities = [0.1, 0.5, 1.0, 5.0, 10.0]  
    
    print("=" * 60)
    print(" Starting Cleo Network Simulation Sweep Benchmark")
    print("=" * 60)
    
    results = {}

    for intensity in stimulus_intensities:
        print(f"\n[+] Running network simulation for Intensity = {intensity}...")
        
        start_time = time.time()
        
        # -------------------------------------------------------------
        # TODO: Insert your Cleo network execution call here
        # Example:
        # sim.run(100 * ms)
        # -------------------------------------------------------------
        
        # Temporary placeholder delay to verify script structure
        time.sleep(0.5) 
        
        elapsed_time = time.time() - start_time
        results[intensity] = elapsed_time
        
        print(f"    -> Completed in {elapsed_time:.3f} seconds.")

    print("\n" + "=" * 60)
    print(" SWEEP BENCHMARK SUMMARY")
    print("=" * 60)
    for intensity, exec_time in results.items():
        print(f" Intensity: {intensity:5.1f} | Execution Time: {exec_time:.3f} s")

if __name__ == "__main__":
    benchmark_network_sweep()
