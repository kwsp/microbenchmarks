## Cosine similarity

AVX2 impl from https://ashvardanian.com/posts/understanding-simd-complexity/

### Intel i9-13900K

```
2024-12-01T15:37:21-06:00
Running C:\src\microbenches\build\cl\src\similarity\Release\similarity_benchmarks.exe
Run on (16 X 4200 MHz CPU s)
CPU Caches:
  L1 Data 32 KiB (x8)
  L1 Instruction 32 KiB (x8)
  L2 Unified 1024 KiB (x8)
  L3 Unified 98304 KiB (x1)
---------------------------------------------------------------------
Benchmark                           Time             CPU   Iterations
---------------------------------------------------------------------
BM_similarity_naive/4096         2470 ns         2288 ns       280000
BM_similarity_naive/16382        9887 ns         9835 ns        74667
BM_similarity_avx2/4096           427 ns          424 ns      1659259
BM_similarity_avx2/16382         1717 ns         1726 ns       407273
BM_similarity_Eigen3/4096         458 ns          459 ns      1600000
BM_similarity_Eigen3/16382       2116 ns         2131 ns       344615
BM_similarity_Arma/4096          7613 ns         7324 ns        74667
BM_similarity_Arma/16382        30518 ns        29576 ns        28000
```

### Apple M1

```
Unable to determine clock rate from sysctl: hw.cpufrequency: No such file or directory
This does not affect benchmark measurements, only the metadata output.
***WARNING*** Failed to set thread affinity. Estimated CPU frequency may be incorrect.
2024-12-01T15:46:37-06:00
Running /Users/tnie/code/cpp/microbenchmarks/build/clang/src/similarity/Release/similarity_benchmarks
Run on (8 X 24 MHz CPU s)
CPU Caches:
  L1 Data 64 KiB
  L1 Instruction 128 KiB
  L2 Unified 4096 KiB (x8)
Load Average: 4.34, 3.51, 2.73
---------------------------------------------------------------------
Benchmark                           Time             CPU   Iterations
---------------------------------------------------------------------
BM_similarity_naive/4096         5079 ns         5079 ns       136898
BM_similarity_naive/16382       20785 ns        20723 ns        34206
BM_similarity_Eigen3/4096        1388 ns         1378 ns       513746
BM_similarity_Eigen3/16382       5965 ns         5903 ns       120821
BM_similarity_Arma/4096         10109 ns        10096 ns        68879
BM_similarity_Arma/16382        41082 ns        41005 ns        17022
```
