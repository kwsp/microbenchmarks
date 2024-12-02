# Hilbert benchmarks

Original

```
Unable to determine clock rate from sysctl: hw.cpufrequency: No such file or directory
This does not affect benchmark measurements, only the metadata output.
***WARNING*** Failed to set thread affinity. Estimated CPU frequency may be incorrect.
2024-12-02T15:44:34-06:00
Running /Users/tnie/code/cpp/microbenchmarks/build/clang/src/hilbert/RelWithDebInfo/hilbert_benchmarks
Run on (8 X 24 MHz CPU s)
CPU Caches:
  L1 Data 64 KiB
  L1 Instruction 128 KiB
  L2 Unified 4096 KiB (x8)
Load Average: 2.59, 2.38, 2.29
----------------------------------------------------------------------
Benchmark                            Time             CPU   Iterations
----------------------------------------------------------------------
BM_hilbert_fftw<float>/2048      16709 ns        16695 ns        41709
BM_hilbert_fftw<float>/4096      37182 ns        37141 ns        18797
BM_hilbert_fftw<float>/8192      78608 ns        78545 ns         8851
```

Manual complex abs

```
Unable to determine clock rate from sysctl: hw.cpufrequency: No such file or directory
This does not affect benchmark measurements, only the metadata output.
***WARNING*** Failed to set thread affinity. Estimated CPU frequency may be incorrect.
2024-12-02T15:56:53-06:00
Running /Users/tnie/code/cpp/microbenchmarks/build/clang/src/hilbert/RelWithDebInfo/hilbert_benchmarks
Run on (8 X 24 MHz CPU s)
CPU Caches:
  L1 Data 64 KiB
  L1 Instruction 128 KiB
  L2 Unified 4096 KiB (x8)
Load Average: 2.46, 3.28, 2.92
----------------------------------------------------------------------
Benchmark                            Time             CPU   Iterations
----------------------------------------------------------------------
BM_hilbert_fftw<float>/2048      14644 ns        14624 ns        47620
BM_hilbert_fftw<float>/4096      32878 ns        32826 ns        21266
BM_hilbert_fftw<float>/8192      70425 ns        70419 ns         9947
```
