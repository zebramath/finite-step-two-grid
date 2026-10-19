# Finite-step two-grid interpolation

Companion C++ code for **Early Stopping in PCG-Based Interpolation for Two-Grid Solvers**, by **Zhengping Cui** and **Jun Hu**.

Both authors: School of Mathematical Sciences, Peking University, Beijing, China. Jun Hu is also affiliated with the Chongqing Research Institute of Big Data, Peking University, Chongqing, China.

## Contents

- `src/`: core algorithms and diffusion discretization.
- `experiments/`: five experiment drivers with fixed parameters and seeds.
- `results/`: generated CSV files; initially empty.

## Build and run

Requires CMake 3.16+ and a C++17 compiler with OpenMP and GCC-compatible options. Run from the repository root in a Linux shell:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j2
./build/experiment1_finite_path_comparison
./build/experiment2_spectral_path
./build/experiment4_local_diagnostic_export
./build/experiment5_robustness
./build/experiment6_endpoint_comparison
```

Each program writes `results/experimentN.csv`; experiment 4 exports matrices for subsequent analysis. Existing output files are overwritten. Timing values are in seconds and depend on the machine.

`TGI_THREADS` sets the thread count (default: 4). `TGI_RESULTS_DIR` overrides the output directory, which is created automatically.

Please cite the accompanying paper when using this code. Contact: [Zhengping Cui](mailto:2400010714@stu.pku.edu.cn) or [Jun Hu](mailto:hujun@math.pku.edu.cn).
