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
