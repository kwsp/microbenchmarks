# Hilbert benchmarks

### Intel i9-13900K

```
Running C:\src\microbenches\build\cl\src\hilbert\Release\hilbert_benchmarks.exe
Run on (32 X 2995 MHz CPU s)
CPU Caches:
  L1 Data 48 KiB (x16)
  L1 Instruction 32 KiB (x16)
  L2 Unified 2048 KiB (x16)
  L3 Unified 36864 KiB (x1)
-----------------------------------------------------------------------------
Benchmark                                   Time             CPU   Iterations
-----------------------------------------------------------------------------
BM_hilbert_fftw<float>/2048              5639 ns         4046 ns       224000
BM_hilbert_fftw<float>/4096             13807 ns         9531 ns       100000
BM_hilbert_fftw<float>/8192             41051 ns        27902 ns        28000
BM_hilbert_fftw<double>/2048            11111 ns         7394 ns       112000
BM_hilbert_fftw<double>/4096            32381 ns        21031 ns        49778
BM_hilbert_fftw<double>/8192            76107 ns        47083 ns        16593
BM_hilbert_fftw_split<float>/2048        3856 ns         2679 ns       560000
BM_hilbert_fftw_split<float>/4096       12018 ns         7500 ns       100000
BM_hilbert_fftw_split<float>/8192       33005 ns        23856 ns        37333
BM_hilbert_fftw_split<double>/2048       8068 ns         6250 ns       100000
BM_hilbert_fftw_split<double>/4096      29233 ns        19252 ns        37333
BM_hilbert_fftw_split<double>/8192      68246 ns        39899 ns        20364
```

### Apple M1

```
Unable to determine clock rate from sysctl: hw.cpufrequency: No such file or directory
This does not affect benchmark measurements, only the metadata output.
***WARNING*** Failed to set thread affinity. Estimated CPU frequency may be incorrect.
2024-12-02T23:57:30-06:00
Running /Users/tnie/code/cpp/microbenchmarks/build/clang/src/hilbert/Release/hilbert_benchmarks
Run on (8 X 24 MHz CPU s)
CPU Caches:
  L1 Data 64 KiB
  L1 Instruction 128 KiB
  L2 Unified 4096 KiB (x8)
Load Average: 1.73, 1.57, 1.58
-----------------------------------------------------------------------------
Benchmark                                   Time             CPU   Iterations
-----------------------------------------------------------------------------
BM_hilbert_fftw<float>/2048             14676 ns        14668 ns        48033
BM_hilbert_fftw<float>/4096             33065 ns        33064 ns        21210
BM_hilbert_fftw<float>/8192             70893 ns        70849 ns         9894
BM_hilbert_fftw<double>/2048            15538 ns        15533 ns        44919
BM_hilbert_fftw<double>/4096            35428 ns        35426 ns        19727
BM_hilbert_fftw<double>/8192            77784 ns        77780 ns         9052
BM_hilbert_fftw_split<float>/2048       14938 ns        14938 ns        46720
BM_hilbert_fftw_split<float>/4096       34372 ns        34370 ns        20582
BM_hilbert_fftw_split<float>/8192       74966 ns        74149 ns         9809
BM_hilbert_fftw_split<double>/2048      16273 ns        16273 ns        43062
BM_hilbert_fftw_split<double>/4096      36966 ns        36965 ns        18971
BM_hilbert_fftw_split<double>/8192      80351 ns        80346 ns         8685
```
