# Matrix Conversion Benchmark

Intel i9-13900K, Windows 11. MSVC cannot vectorize the loops.

```
2024-06-25T22:16:29-05:00
Running C:\Users\tnie\code\microbenches\build\win64\src\matrix_conversion\Release\matrix_conversion_benchmark.exe
Run on (32 X 2995 MHz CPU s)
CPU Caches:
  L1 Data 48 KiB (x16)
  L1 Instruction 32 KiB (x16)
  L2 Unified 2048 KiB (x16)
  L3 Unified 36864 KiB (x1)
------------------------------------------------------------------------------
Benchmark                                    Time             CPU   Iterations
------------------------------------------------------------------------------
BM_ArmadilloConversion/256               13242 ns         9267 ns        80929
BM_ArmadilloConversion/512              365968 ns       258092 ns         2240
BM_ArmadilloConversion/4096           24568550 ns     17708333 ns           30
BM_HandRolledConversion/256              20174 ns        12556 ns        56000
BM_HandRolledConversion/512              80300 ns        50000 ns        10000
BM_HandRolledConversion/4096           9759993 ns      6944444 ns           90
BM_OpenMPConversion/256                   5543 ns         5625 ns       100000
BM_OpenMPConversion/512                  13345 ns        12870 ns        49778
BM_OpenMPConversion/4096               4303319 ns      5000000 ns          100
BM_OpenCVParallelConversion1/256         20355 ns        19496 ns        34462
BM_OpenCVParallelConversion1/512         29814 ns        30134 ns        28000
BM_OpenCVParallelConversion1/4096      3527950 ns      3244174 ns          236
BM_OpenCVParallelConversion2/256         10447 ns        11475 ns        64000
BM_OpenCVParallelConversion2/512         18806 ns        18032 ns        40727
BM_OpenCVParallelConversion2/4096      3501157 ns      3216912 ns          204
BM_OpenCVMatConversion/256                6702 ns         4653 ns       154483
BM_OpenCVMatConversion/512               39179 ns        27344 ns        28000
BM_OpenCVMatConversion/4096            8680204 ns      4823826 ns          149
BM_OpenCVMatParallelConversion/256       10115 ns        10045 ns        74667
BM_OpenCVMatParallelConversion/512       18062 ns        18032 ns        40727
BM_OpenCVMatParallelConversion/4096    3422001 ns      3125000 ns          280
```

Apple Macbook Air M1. Clang is able to vectorize the handrolled loops here.

```
Running /Users/tnie/code/cpp/microbenches/build/release/src/matrix_conversion/matrix_conversion_benchmark
Run on (8 X 24 MHz CPU s)
CPU Caches:
  L1 Data 64 KiB
  L1 Instruction 128 KiB
  L2 Unified 4096 KiB (x8)
Load Average: 4.77, 40.19, 43.64
------------------------------------------------------------------------------
Benchmark                                    Time             CPU   Iterations
------------------------------------------------------------------------------
BM_ArmadilloConversion/256               10562 ns        10518 ns        66128
BM_ArmadilloConversion/512              124011 ns       123235 ns         5650
BM_ArmadilloConversion/4096           12081034 ns     11013613 ns           62
BM_HandRolledConversion/256               6965 ns         6949 ns        99691
BM_HandRolledConversion/512              34383 ns        34285 ns        21606
BM_HandRolledConversion/4096           3727255 ns      3697326 ns          187
BM_OpenCVParallelConversion1/256       3389555 ns      3180462 ns          212
BM_OpenCVParallelConversion1/512      14009253 ns     12099855 ns           55
BM_OpenCVParallelConversion1/4096    896351000 ns    802086000 ns            1
BM_OpenCVParallelConversion2/256         17096 ns        16924 ns        41771
BM_OpenCVParallelConversion2/512         40472 ns        40144 ns        17431
BM_OpenCVParallelConversion2/4096      3945687 ns      3744016 ns          187
BM_OpenCVMatConversion/256                7730 ns         7693 ns        89194
BM_OpenCVMatConversion/512               35324 ns        35219 ns        22661
BM_OpenCVMatConversion/4096            2972343 ns      2965856 ns          236
BM_OpenCVMatParallelConversion/256       16597 ns        16468 ns        42360
BM_OpenCVMatParallelConversion/512       46563 ns        46185 ns        15265
BM_OpenCVMatParallelConversion/4096    4058611 ns      3365724 ns          185
```
