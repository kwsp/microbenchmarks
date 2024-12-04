# %%
from gbenchutils import load_benchmarks_from_default_build_dir
import gbenchutils


context, benchmarks, xdata = load_benchmarks_from_default_build_dir("output.json")

benchmarks.head()


# %%
from tqdm import tqdm
import numpy as np
import timeit


def measure_throughput_np(func: callable, N: int | list[int], iterations=10000):
    """
    func takes a single arg (np.array of size N)
    """
    if isinstance(N, (int, np.integer)):
        arr = np.random.random(N)
        func(arr)
        time = timeit.timeit(lambda: func(arr), number=iterations) / iterations
        throughput = N / time
        return throughput

    return np.array([measure_throughput_np(func, n) for n in tqdm(N)])


# %%
from scipy import signal


def np_sp_hilbert(x):
    return np.abs(signal.hilbert(x))


np_throughput = measure_throughput_np(np_sp_hilbert, xdata)
np_throughput


# %%
import importlib

importlib.reload(gbenchutils)

pylib_throughputs = [
    (f"Numpy {np.__version__}", np_throughput),
]

replace_name = {}

fig = gbenchutils.plot_throughputs_bar(
    benchmarks[benchmarks["template_param_1"] == "double"],
    replace_name=replace_name,
    highlight_name="fftw",
    title="Hilbert",
    pylib_throughputs=pylib_throughputs,
    xlim=200,
)

# %%
