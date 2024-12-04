# Matrix Shift

```
2024-12-04T14:57:27-06:00
Running /Users/tnie/code/cpp/microbenchmarks/build/clang/src/matrix_shift/Release/matrix_shift_benchmark
Run on (8 X 24 MHz CPU s)
CPU Caches:
  L1 Data 64 KiB
  L1 Instruction 128 KiB
  L2 Unified 4096 KiB (x8)
Load Average: 2.49, 2.37, 2.48
-------------------------------------------------------------------
Benchmark                         Time             CPU   Iterations
-------------------------------------------------------------------
BM_Shift/256                   4466 ns         4212 ns       169507
BM_Shift/512                  31036 ns        26152 ns        28211
BM_Shift/4096               4101214 ns      4098667 ns          162
BM_FastShiftColumns/256        9170 ns         9029 ns        80539
BM_FastShiftColumns/512       31768 ns        31767 ns        20107
BM_FastShiftColumns/4096    7591415 ns      7571886 ns           88
```
