# %%
from pprint import pprint
import json
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import cpuinfo

cpu_info = cpuinfo.get_cpu_info()
cpu: str = cpu_info["brand_raw"]
print(cpu)


def benchmark_parse_parameters(benchmark: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a df with the Benchmark parameters parse into separate columns:

    func_name
    param_1
    param_2
    ...
    throughput
    """
    benchmark = benchmark.copy()
    parameters_list = benchmark.run_name.str.split("/").to_list()

    # The first value is the same as run_name. Discard
    n_params = len(parameters_list[0]) - 1

    print(f"Found {n_params=}")

    benchmark["func_name"] = [
        parameters[0].split("BM_", 1)[1] for parameters in parameters_list
    ]
    for i in range(1, n_params + 1):
        benchmark[f"param_{i}"] = [int(parameters[i]) for parameters in parameters_list]

    # Throughput [samples/s]
    benchmark["throughput"] = (benchmark.param_1) / (benchmark.real_time * 1e-9)

    return benchmark


if os.name == "nt":
    path = "./../../build/cl/src/conv1d/Release/output_dense.json"
else:
    path = "../../build/clang/src/conv1d/Release/output_dense.json"

with open(path, "r") as fp:
    d = json.load(fp)

pprint(d["context"])
print()

benchmarks = pd.DataFrame(d["benchmarks"])
benchmarks = benchmark_parse_parameters(benchmarks)
benchmarks.head()

# %%
bar_width = 0.2
x = np.arange(len(benchmarks["param_1"].unique()))
benchmark_groups = {name: df for name, df in benchmarks.groupby("func_name")}

# # Desired key order
# keys = [
#     # "BM_conv1d_ipp_full_direct<double>",
#     "conv1d_Arma<double>",
#     "conv1d_ipp_full_fft<double>",
#     "conv1d_fftconv_oa<double>",
# ]
# labels = [
#     # "ipp (direct)",
#     "Armadillo",
#     "IPP (FFT)",
#     "fftconv",
# ]
# benchmark_groups = {k: benchmark_groups[k] for k in keys}

colors = []
# labels = keys

param_2 = 245


# %%
benchmarks["param_1"].unique()

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


xdata = benchmarks["param_1"].unique()

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
names, throughputs_mean, throughputs_std = [], [], []
for i, (name, df) in enumerate(benchmark_groups.items()):
    df = df[df.param_2 == param_2]
    name = name.split("_", 1)[1].split("<", 1)[0]
    throughput = df.throughput.array * 1e-6  # MS/s

    names.append(name)
    throughputs_mean.append(np.mean(throughput))
    throughputs_std.append(np.std(throughput))

torch_version = torch.__version__.split("+")[0]
python_lib_throughputs = [
    (f"Numpy {np.__version__}", numpy_throughputs),
    (f"Scipy {sp.__version__}", scipy_throughputs),
    (f"Scipy {sp.__version__} (fft)", scipy_fft_throughputs),
    (f"Scipy {sp.__version__} (oa)", scipy_oa_throughputs),
    (f"PyTorch {torch_version}", torch_throughputs),
]

for name, throughput in python_lib_throughputs:
    throughput = np.array(throughput) * 1e-6

    names.append(name)
    throughputs_mean.append(np.mean(throughput))
    throughputs_std.append(np.std(throughput))


REPLACE = {
    "fftconv_oa": "fftconv",
    "ipp_full_fft": "Intel IPP (FFT)",
    "ipp_full_direct": "Intel IPP (direct)",
    "Arma": "Armadillo",
    "Eigen": "Eigen3",
    "BLAS": "BLAS (im2col)",
    "Accelerate_vDSP": "Apple Accelerate",
}


def replace_name(name: str):
    ret = REPLACE.get(name)
    return ret if ret else name


names = [replace_name(name) for name in names]

names = np.array(names)
throughputs_mean = np.array(throughputs_mean)
throughputs_std = np.array(throughputs_std)

idx = np.argsort(throughputs_mean)
throughputs_mean = throughputs_mean[idx]
throughputs_std = throughputs_std[idx]
names = names[idx]

f, ax = plt.subplots()
x = np.arange(len(names))
rects = ax.barh(x, throughputs_mean, xerr=throughputs_std)
ax.bar_label(rects, fmt="%.1f")

fftconv_idx = np.argmax(names == "fftconv")
rects[fftconv_idx].set_color("tab:green")

ax.set_yticks(x, labels=names)
ax.tick_params(left=False)
ax.set_xlabel("Throughput (MS/s)")
ax.set_xlim(None, 250)
title = f"CPU 1D Convolution\nN = {xdata.tolist()}\nk = {param_2}"
title = cpu + "\n" + title
ax.set_title(title)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
f.tight_layout()
f.savefig(f"Conv1d Throughput Bar (k={param_2}) {cpu}.svg")


# %%
f, ax = plt.subplots()

# for i, (name, df) in enumerate(benchmark_groups.items()):
#     df = df[df["param_2"] == param_2]
#     xdata = x + i * bar_width  # bar position
#     # ydata = df["real_time"] / 1000
#     ydata = df["throughput"] * 1e-6
#     # label = labels[i]
#     label = name.split("_", 1)[1].split("<", 1)[0]
#     ax.bar(xdata, ydata, width=bar_width, label=label)
#     # ax.bar(xdata, ydata, width=bar_width, label=name)
# ax.set_xticks(x + (len(benchmark_groups) - 1) / 2 * bar_width)
# ax.set_xticklabels(df["param_1"])

for i, (name, df) in enumerate(benchmark_groups.items()):
    df = df[df["param_2"] == param_2]
    xdata = df["param_1"]
    # ydata = df["real_time"] / 1000
    ydata = df["throughput"] * 1e-6
    # label = labels[i]
    name = name.split("_", 1)[1].split("<", 1)[0]
    label = replace_name(name)
    ax.scatter(xdata, ydata, label=label, marker="x")
    ax.plot(xdata, ydata)

for name, throughputs in python_lib_throughputs:
    ydata = throughputs * 1e-6
    ax.scatter(xdata, ydata, label=name, marker="x")
    ax.plot(xdata, ydata)


# ax.set_ylabel("Real time ($\mu s$)")
ax.set_ylabel("Throughput (MS / s)")
# ax.set_yscale("log")

title = f"Convolution (kernel size = {param_2})"
title = cpu + "\n" + title
ax.set_title(title)
ax.set_xlabel("N")
f.tight_layout()
f.savefig(f"dense2_nolegend {cpu}.png")

idx_rev = idx[::-1]
handles, labels = ax.get_legend_handles_labels()
ax.legend([handles[i] for i in idx_rev], [labels[i] for i in idx_rev])

f.savefig(f"Conv1d Throughput Line (k={param_2}) {cpu}.svg")


# %%
def calc_improvement(func_name1: str, func_name2: str):
    df1 = benchmark_groups[func_name1]
    df2 = benchmark_groups[func_name2]
    return df1["real_time"].array / df2["real_time"].array


# calc_improvement(
#     "BM_conv1d_ipp_full_fft<double>",
#     "BM_conv1d_fftconv_oa<double>",
# ).mean()

baseline_alg = "conv1d_ipp_full_direct<double>"
faster_algs = [
    "conv1d_Eigen<double>",
    "conv1d_KFR<double>",
    "conv1d_OpenCV<double>",
    "conv1d_ipp_full_fft<double>",
    "conv1d_fftconv_oa<double>",
]

f, ax = plt.subplots()
for i, faster_alg in enumerate(faster_algs):
    speedups = calc_improvement(baseline_alg, faster_alg)
    print(faster_alg, speedups.mean())
    ax.plot(speedups, label=faster_alg)

ax.set_xlabel("N")
ax.legend()
f.tight_layout()


# %%
from scipy.interpolate import griddata

# %matplotlib qt


def plot_surface(func_name):
    from matplotlib import cm

    df = benchmark_groups[func_name]

    x = df.param_1.array
    y = df.param_2.array
    z = df.real_time.array

    USE_INTERP = True
    if USE_INTERP:
        xi = np.linspace(x.min(), x.max(), 50)
        yi = np.linspace(y.min(), y.max(), 50)
        xi, yi = np.meshgrid(xi, yi)

        zi = griddata((x, y), z, (xi, yi), method="cubic")

    else:
        # Don't interpolate.
        counts = y.value_counts().array
        count = counts[0]
        assert np.all(count == counts)

        xsize = count
        ysize = x.size // xsize
        xi = x.reshape((xsize, ysize))
        yi = y.reshape((xsize, ysize))
        zi = z.reshape((xsize, ysize))

    f, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(6, 6))
    ax.plot_surface(xi, yi, zi, cmap=cm.coolwarm)
    ax.set_xlabel("Signal size (samples)")
    ax.set_ylabel("Kernel size")
    ax.set_zlabel("Real time (us)")
    f.tight_layout()


plot_surface(faster_algs[-1])

# %%
