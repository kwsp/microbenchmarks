# %%
import gbenchutils

context, benchmarks, xdata = gbenchutils.load_benchmarks_from_default_build_dir(
    "output.json"
)
param_2 = 165


# %%
from tqdm import tqdm
import numpy as np
import timeit


def measure_func_throughput_np(
    func: callable, N: int | list[int], k: int, iterations=10000
):
    if isinstance(N, (int, np.integer)):
        kernel = np.random.random(k)
        arr = np.random.random(N)
        func(arr, kernel)
        time = timeit.timeit(lambda: func(arr, kernel), number=iterations) / iterations
        throughput = N / time
        return throughput
    else:
        return np.array([measure_func_throughput_np(func, n, param_2) for n in tqdm(N)])


# %%
numpy_throughputs = measure_func_throughput_np(np.convolve, xdata, param_2)
numpy_throughputs

# %%
import scipy as sp
from scipy import signal

scipy_throughputs = measure_func_throughput_np(signal.convolve, xdata, param_2)
scipy_throughputs

# %%
scipy_fft_throughputs = measure_func_throughput_np(signal.fftconvolve, xdata, param_2)
scipy_fft_throughputs

# %%
scipy_oa_throughputs = measure_func_throughput_np(signal.oaconvolve, xdata, param_2)
scipy_oa_throughputs

# %%
import torch
import torch.nn.functional as F


def measure_func_throughput_torch(func: callable, N: int, k: int, iterations=10000):
    if isinstance(N, (int, np.integer)):
        kernel = torch.rand((1, 1, k), dtype=torch.float64)
        arr = torch.rand((1, 1, N), dtype=torch.float64)
        with torch.no_grad():
            time = (
                timeit.timeit(lambda: func(arr, kernel), number=iterations) / iterations
            )
        throughput = N / time
        return throughput
    else:
        return np.array(
            [measure_func_throughput_torch(func, n, param_2) for n in tqdm(N)]
        )


torch_throughputs = measure_func_throughput_torch(F.conv1d, xdata, param_2)
torch_throughputs


# %%

torch_version = torch.__version__.split("+")[0]
python_lib_throughputs = [
    (f"Numpy {np.__version__}", numpy_throughputs),
    (f"Scipy {sp.__version__}", scipy_throughputs),
    (f"Scipy {sp.__version__} (fft)", scipy_fft_throughputs),
    (f"Scipy {sp.__version__} (oa)", scipy_oa_throughputs),
    (f"PyTorch {torch_version}", torch_throughputs),
]


REPLACE = {
    "fftconv_oa": "fftconv",
    "ipp_full_fft": "Intel IPP (FFT)",
    "ipp_full_direct": "Intel IPP (direct)",
    "Arma": "Armadillo",
    "Eigen": "Eigen3",
    "BLAS": "BLAS (im2col)",
    "Accelerate_vDSP": "Apple Accelerate",
}


gbenchutils.plot_throughputs_bar(
    benchmarks,
    param_2,
    title="conv1d",
    replace_name=REPLACE,
    highlight_name="fftconv",
    pylib_throughputs=python_lib_throughputs,
    xlim=250,
)
