# Conv1d (CPU)

### Intel i9-13900K

Compiler: MSVC 19.40.33812

Kernel size = 165

<p align="center">
<img src="./plots/conv1d-bar-k=165-13th_gen_intelr_coretm_i9-13900K.svg" width="45%">
<img src="./plots/conv1d-line-k=165-13th_gen_intelr_coretm_i9-13900K.svg" width="45%">
</p>

Kernel size = 245

<p align="center">
<img src="./plots/conv1d-bar-k=245-13th_gen_intelr_coretm_i9-13900K.svg" width="45%">
<img src="./plots/conv1d-line-k=245-13th_gen_intelr_coretm_i9-13900K.svg" width="45%">
</p>

### AMD

Compiler: MSVC 19.40.33812

_Note: This CPU supports AVX512 but FFTW3 built from VCPKG only supports up to AVX2_

Kernel size = 165

<p align="center">
<img src="./plots/conv1d-bar-k=165-amd_ryzen_7_7800x3d.svg" width="45%">
<img src="./plots/conv1d-line-k=165-amd_ryzen_7_7800x3d.svg" width="45%">
</p>

Kernel size = 245

<p align="center">
<img src="./plots/conv1d-bar-k=245-amd_ryzen_7_7800x3d.svg" width="45%">
<img src="./plots/conv1d-line-k=245-amd_ryzen_7_7800x3d.svg" width="45%">
</p>

### Apple M1

Compiler: Apple clang 16.0.0

Kernel size = 165

<p align="center">
<img src="./plots/conv1d-bar-k=165-apple_m1.svg" width="45%">
<img src="./plots/conv1d-line-k=165-apple_m1.svg" width="45%">
</p>

Kernel size = 245

<p align="center">
<img src="./plots/conv1d-bar-k=245-apple_m1.svg" width="45%">
<img src="./plots/conv1d-line-k=245-apple_m1.svg" width="45%">
</p>
