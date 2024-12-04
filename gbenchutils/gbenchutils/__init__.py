from pathlib import Path
from pprint import pprint
import re
import os
import cpuinfo
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

cpu_info = cpuinfo.get_cpu_info()
cpu: str = cpu_info["brand_raw"]
print(cpu)


def get_default_build_dir(out="output.json") -> Path:
    """
    Assumes the current working directory is the source directory, and the project is built with cl on Windows and clang everywhere else.
    """
    if os.name == "nt":
        path = Path() / "../../build/cl/src" / Path.cwd().name / f"Release/{out}"
    else:
        path = Path() / "../../build/clang/src" / Path.cwd().name / f"Release/{out}"

    return path


def load_benchmarks_from_default_build_dir(out="output.json"):
    import json

    path = get_default_build_dir(out)

    with open(path, "r") as fp:
        d = json.load(fp)

    context = d["context"]
    pprint(context)
    print()

    benchmarks = pd.DataFrame(d["benchmarks"])
    benchmarks = benchmark_parse_parameters(benchmarks)
    xdata = benchmarks["param_1"].unique()

    return context, benchmarks, xdata


def benchmark_parse_parameters(benchmark: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a df with the Benchmark parameters (from Google Benchmark) and function template parameters parsed into separate columns:

    func_name: str, original function name
    param_1: int
    param_2: int
    ...
    template_param_1: str
    template_param_2: str
    ...
    throughput: float, items per second
    """
    benchmark = benchmark.copy()

    # Parse benchmark arguments
    parameters_list = benchmark.run_name.str.split("/").to_list()

    # The first value is the same as run_name. Discard
    n_params = len(parameters_list[0]) - 1
    print(f"Found {n_params=}")

    func_names = [parameters[0].split("BM_", 1)[1] for parameters in parameters_list]
    benchmark["func_name"] = func_names

    for i in range(1, n_params + 1):
        benchmark[f"param_{i}"] = [
            (int(params[i]) if i < len(params) else None) for params in parameters_list
        ]

    # Parse template arguments

    def parse_template_params(func_name: str) -> list[str] | None:
        "Given 'some_func<float, double>', returns ['float', 'double']"
        match = re.match(".+<(.+)>", func_name)
        if match:
            captured_tmpl_args = match.groups()[0]
            return captured_tmpl_args.split(", ")

    template_params = [parse_template_params(fname) for fname in func_names]
    n_template_params = max(len(args) for args in template_params)
    print(f"Found {n_template_params=}")
    for i in range(n_template_params):
        benchmark[f"template_param_{i + 1}"] = [
            (params[i] if i < len(params) else None) for params in template_params
        ]

    # Throughput [samples/s]
    benchmark["throughput"] = (benchmark.param_1) / (benchmark.real_time * 1e-9)

    return benchmark


def plot_throughputs_bar(
    benchmarks: pd.DataFrame,
    param_2: int | None = None,
    replace_name: dict[str, str] = {},
    highlight_name: str | None = None,
    title: str | None = None,
    group_column="func_name",
    pylib_throughputs: list[tuple[str, np.ndarray]] = None,
    xlim: int = 600,
    xlabel="Throughput (MS/s)",
):
    """
    Generate a bar chart and line chart of function throughputs from the benchmark output from Google Benchmark.

    `param_1` will be on the x-axis and `func_name` will be on the y-axis. Optionally add more data points with the param `pylib_throughputs`, where the array should contain throughputs over `param_1` for every function provided.

    Params
    ------
        benchmark_groups: pd.DataFrame GroupBy (func_name).
        param_2: Optional second parameter to filter the DataFrame by.
        replace_name: Optional mapping from func_name to a more display friendly name.
        highlight_name: Highlight this function in the bar plot.
        pylib_throughputs: A list of pairs (func_name, throughput array) to add to the plots. Throughput. Items per second
        xlim: Upper range of x-axis
    """
    func_names, throughputs_mean, throughputs_std = [], [], []

    xdata = benchmarks["param_1"].unique()

    # Compute the mean and stds
    for func_name, df in benchmarks.groupby(group_column):
        if param_2:
            df = df[df.param_2 == param_2]
        func_name = func_name.split("_", 1)[1].split("<", 1)[0]
        throughput = df.throughput.array * 1e-6  # MS/s

        func_names.append(func_name)
        throughputs_mean.append(np.mean(throughput))
        throughputs_std.append(np.std(throughput))

    if pylib_throughputs:
        for func_name, throughput in pylib_throughputs:
            throughput = np.array(throughput) * 1e-6

            func_names.append(func_name)
            throughputs_mean.append(np.mean(throughput))
            throughputs_std.append(np.std(throughput))

    func_names = [replace_name.get(name, name) for name in func_names]

    func_names = np.array(func_names)
    throughputs_mean = np.array(throughputs_mean)
    throughputs_std = np.array(throughputs_std)

    idx = np.argsort(throughputs_mean)
    throughputs_mean = throughputs_mean[idx]
    throughputs_std = throughputs_std[idx]
    func_names = func_names[idx]

    ### Horizontal Bar plot
    f, ax = plt.subplots()
    x = np.arange(len(func_names))
    rects = ax.barh(x, throughputs_mean, xerr=throughputs_std)
    ax.bar_label(rects, fmt="%.1f")

    if highlight_name is not None:
        fftconv_idx = np.argmax(func_names == highlight_name)
        rects[fftconv_idx].set_color("tab:green")

    ax.set_yticks(x, labels=func_names)
    ax.tick_params(left=False)
    ax.set_xlabel(xlabel)
    ax.set_xlim(None, xlim)

    if title:  # provided in param
        title_parts = [cpu, title, f"N = {xdata.tolist()}"]
    else:
        title_parts = [cpu, f"N = {xdata.tolist()}\n"]

    if param_2 is not None:
        title_parts.append(f"k = {param_2}")

    plot_title = "\n".join(title_parts)
    ax.set_title(plot_title)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    f.tight_layout()

    if param_2:
        filename = f"Throughput Bar (k={param_2}) {cpu}.svg"
    else:
        filename = f"Throughput Bar {cpu}.svg"
    f.savefig(filename)
    print(f"Bar plot saved to {filename}")

    ### Plot  throughputs line
    f, ax = plt.subplots()

    for name, df in benchmarks.groupby(group_column):
        if param_2:
            df = df[df["param_2"] == param_2]
        xdata = df["param_1"]
        ydata = df["throughput"] * 1e-6
        name = name.split("_", 1)[1].split("<", 1)[0]
        label = replace_name.get(name, name)
        ax.scatter(xdata, ydata, label=label, marker="x")
        ax.plot(xdata, ydata)

    for name, throughputs in pylib_throughputs:
        ydata = throughputs * 1e-6
        ax.scatter(xdata, ydata, label=name, marker="x")
        ax.plot(xdata, ydata)

    ax.set_ylabel(xlabel)
    # ax.set_yscale("log")

    title_parts = [cpu]
    if title:
        title_parts.append(title)

    if param_2:
        title_parts.append(f"k = {param_2}")

    title = "\n".join(title_parts)
    ax.set_title(title)
    ax.set_xlabel("N")
    f.tight_layout()
    f.savefig(f"dense2_nolegend {cpu}.png")

    idx_rev = idx[::-1]
    handles, labels = ax.get_legend_handles_labels()
    ax.legend([handles[i] for i in idx_rev], [labels[i] for i in idx_rev])

    if param_2:
        fname = f"Throughput Line (k={param_2}) {cpu}.svg"
    else:
        fname = f"Throughput Line {cpu}.svg"
    f.savefig(fname)
    print(f"Line plot saved to {fname}")
