# Matrix Shift

```
Running /Users/tnie/code/cpp/microbenches/build/clang-release/src/matrix_shift/matrix_shift_benchmark
Run on (8 X 24 MHz CPU s)
CPU Caches:
  L1 Data 64 KiB
  L1 Instruction 128 KiB
  L2 Unified 4096 KiB (x8)
Load Average: 2.60, 2.64, 2.92
---------------------------------------------------------------
Benchmark                     Time             CPU   Iterations
---------------------------------------------------------------
BM_Shift/256               4176 ns         4157 ns       168348
BM_Shift/512              18517 ns        18367 ns        38021
BM_Shift/4096           4231594 ns      4199884 ns          164
BM_ShiftInplace/256        4187 ns         4162 ns       168805
BM_ShiftInplace/512       19576 ns        19429 ns        42230
BM_ShiftInplace/4096    4240984 ns      4218187 ns          160
```
