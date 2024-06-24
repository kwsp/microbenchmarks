# Matrix Conversion Benchmark

Intel i9-13900K, Windows 11

```
2024-06-24T15:44:12-05:00
Running C:\Users\tnie\code\microbenches\build\win64\src\matrix_conversion\Release\matrix_conversion_benchmark.exe
Run on (32 X 2995 MHz CPU s)
CPU Caches:
  L1 Data 48 KiB (x16)
  L1 Instruction 32 KiB (x16)
  L2 Unified 2048 KiB (x16)
  L3 Unified 36864 KiB (x1)
---------------------------------------------------------------------------
Benchmark                                 Time             CPU   Iterations
---------------------------------------------------------------------------
BM_ArmadilloConversion/256            13993 ns         6094 ns       100000
BM_ArmadilloConversion/512           335702 ns       212054 ns         5600
BM_ArmadilloConversion/4096        21692453 ns     17463235 ns           34
BM_HandRolledConversion/256           18689 ns        10463 ns        89600
BM_HandRolledConversion/512           77106 ns        49734 ns        14452
BM_HandRolledConversion/4096        9601958 ns      4799107 ns          140
BM_OpenMPConversion/256                3682 ns         3617 ns       263529
BM_OpenMPConversion/512               10798 ns         9068 ns        89600
BM_OpenMPConversion/4096            4219969 ns      3676471 ns          187
BM_OpenCVParallelConversion/256       21258 ns        18589 ns        34462
BM_OpenCVParallelConversion/512       30113 ns        27274 ns        26353
BM_OpenCVParallelConversion/4096    3538342 ns      3138951 ns          224
```
