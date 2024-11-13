# Conv1d (CPU)

**Intel i9-13900K**

```
2024-11-12T18:18:03-06:00
Running C:\src\microbenches\build\cl\src\conv1d\Release\conv1d_benchmarks.exe
Run on (32 X 2995 MHz CPU s)
CPU Caches:
  L1 Data 48 KiB (x16)
  L1 Instruction 32 KiB (x16)
  L2 Unified 2048 KiB (x16)
  L3 Unified 36864 KiB (x1)
------------------------------------------------------------------------------------
Benchmark                                          Time             CPU   Iterations
------------------------------------------------------------------------------------
BM_conv1d_naive<double>/4096/65               308373 ns       111607 ns         4480
BM_conv1d_BLAS<double>/4096/65                120642 ns        69054 ns         7467
BM_conv1d_Arma<double>/4096/65                 72646 ns        30762 ns        21333
BM_conv1d_Eigen<double>/4096/65                30079 ns        12905 ns        44800
BM_conv1d_KFR<double>/4096/65                 313442 ns       138105 ns         4073
BM_conv1d_OpenCV<double>/4096/65               21737 ns         6562 ns       100000
BM_conv1d_OpenCV_intrin<double>/4096/65        50969 ns        32273 ns        45510
BM_conv1d_fftconv<double>/4096/65              22759 ns        10672 ns        74667
BM_conv1d_fftconv_oa<double>/4096/65            9849 ns         5000 ns       100000
BM_conv1d_fftconv_oa_same<double>/4096/65      10183 ns         4785 ns       160000
```
