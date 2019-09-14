from pathlib import Path
from typing import *
from enum import Enum
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import typing
import operator
import functools
from scipy.stats.mstats import gmean
from pathlib import Path


def _normalize_values(row: pd.Series, baseline_medians: pd.DataFrame):
    progname = str(row.progname)
    for k, v in row.iteritems():
        if not isinstance(v, str):
            row[k] = row[k] / baseline_medians.loc[progname][k]
    return row


def _load_statcounters_csv(csv: Union[Path, Iterable[Path]], metrics: List[str],
                           preprocess_data: Callable[["pd.DataFrame", str], pd.DataFrame], name: str) -> pd.DataFrame:
    if isinstance(csv, Path):
        df = pd.read_csv(csv)
    else:
        assert len(csv) > 0
        df = pd.concat(pd.read_csv(c) for c in csv)
    # add l2cache_misses stat
    df = df.assign(l2cache_misses=lambda x: x.l2cache_read_miss + x.l2cache_write_miss)

    if metrics is None:
        return df
    if preprocess_data is not None:
        df = preprocess_data(df, name)
    return df[["progname"] + metrics]


def progname_mapping_olden(n):
    return {
        "treeadd 21 10 0": "treeadd",
        "perimeter 10 0": "perimeter",
        "mst 1024 0": "mst",
        "bisort 250000 0": "bisort",
        "bisort 250000 0_alloc": "bisort (alloc)",
        "bisort 250000 0_exec": "bisort (exec)",
        "treeadd 21 10 0_alloc": "treeadd (alloc)",
        "treeadd 21 10 0_exec": "treeadd (exec)",
        "perimeter 10 0_alloc": "perimeter (alloc)",
        "perimeter 10 0_exec": "perimeter (exec)",
        "mst 1024 0_alloc": "mst (alloc)",
        "mst 1024 0_exec": "mst (exec)"
    }.get(n, n)


def progname_mapping_mibench(name):
    return name.replace("automotive-", "auto-").replace("-encode", "-enc").replace("-decode", "-dec")


def generate_hardware_results_csv(files: Dict[str, typing.Union[Path, Iterable[Path]]], output_file: Path,
                                  progname_mapping: Callable[[str], str] = None):
    dfs = []
    for k, v in files.items():
        df = _load_statcounters_csv(v, metrics=None)
        df.insert(0, 'target-arch-cpu', k)
        # old analysis script expects these values
        df.insert(0, 'sdk-cpu', "cheri128")
        df.insert(0, 'bitfile-cpu', "cheri128")
        df.insert(0, 'table-struct', "0_256")
        df = df.assign(l2cache_misses=lambda x: x.l2cache_read_miss + x.l2cache_write_miss)
        del df['archname']
        dfs.append(df)
    df = pd.concat(dfs)
    if progname_mapping is not None:
        df["progname"] = df["progname"].apply(progname_mapping)
    df.sort_values(["bitfile-cpu", "sdk-cpu", "target-arch-cpu", "table-struct", "progname"], inplace=True)
    df.to_csv(str(output_file))


def _default_metric_mapping(m: str, variant: str):
    return {
        "cycles": "CPU cycles",
        "instructions": "Instructions",
        "inst_user": "Instructions (userspace)",
        "inst_kernel": "Instructions (kernel)",
        "l2cache_misses": "L2-cache misses",
        "l2cache_flits": "L2-cache flits",
    }.get(m, m)


def _default_metric_mapping_with_variant(m: str, variant: str):
    return variant + " " + _default_metric_mapping(m, variant)


def _default_progname_mapping(n):
    return n


class SummaryBar(Enum):
    ArithmeticMean = ("arith. mean", np.mean)
    # For geomean all values must be greater 1:
    GeometricMean = ("geom. mean", lambda overheads: gmean([x + 1 for x in overheads]) - 1)
    Median = ("median", np.median)

    def __init__(self, label: str, func: Callable[[Iterable[float]], float]):
        self.label = label
        self.func = func


class _RelativeBenchmarkData:
    def __init__(self, values: pd.Series, metric: str, program: str):
        self.program = program
        self.metric = metric
        self.values = values
        # self.data = data
        # subtract one to get a relative overhead
        self.median = np.median(values) - 1
        self.p75 = np.percentile(values, 75) - 1
        self.p25 = np.percentile(values, 25) - 1


class BarResults:
    def __init__(self, data: Mapping[str, pd.Series], metric: str, raw_metric: str):
        self.benchmark_data = []
        self.human_metric = metric
        self.raw_metric = raw_metric
        for k, v in data.items():
            self.benchmark_data.append(_RelativeBenchmarkData(v, raw_metric, k))
        self.medians = np.array([x.median for x in self.benchmark_data])
        self.p75s = np.array([x.p75 for x in self.benchmark_data])
        self.p25s = np.array([x.p25 for x in self.benchmark_data])

    def error_bars(self, error_y: go.bar.ErrorY):
        data_args = dict(type='data', symmetric=False, array=self.p75s - self.medians,
                         arrayminus=self.medians - self.p25s)
        if error_y is None:
            error_y = go.bar.ErrorY()
        error_y.update(data_args)
        # Some default arguments:
        if error_y.thickness is None:
            error_y.thickness = 1
        # FIXME: should report as bug upstream
        # Smaller error bars if there are more than 10 benchmarks:
        if error_y.width is None and len(self.benchmark_data) > 10:
            error_y.width = 2
        return error_y

    def create_bar(self, text_in_bar: bool, error_bar_args: go.bar.ErrorY,
                   add_summary_bars: List[SummaryBar], customize_bar: Callable[["BarResults", go.Bar], go.Bar] = None):
        extra_bars = dict()
        x_values = [x.program for x in self.benchmark_data]
        y_values = list(self.medians.tolist())
        for bar in add_summary_bars:
            print("ADDING", bar.label, bar.func(self.medians))
            x_values.append(bar.label)
            y_values.append(bar.func(self.medians))
            # Note: no error bar here!

        x = go.Bar(
            name=self.human_metric,
            x=x_values, y=y_values,
            error_y=self.error_bars(error_bar_args),
            text=["{0:.2f}%".format(x * 100) for x in y_values] if text_in_bar else None,
            # textfont=dict(size=18),
            textposition='auto',
        )
        if customize_bar is not None:
            x = customize_bar(self, x)
        return x


def reduce_saturation(colour, howmuch):
    # from matplotlib.colors import to_rgb, rgb_to_hsv, hsv_to_rgb, to_hex
    import matplotlib.colors
    # create a less saturated version of the colour:
    rgb = matplotlib.colors.to_rgb(colour)
    hsv = matplotlib.colors.rgb_to_hsv(rgb)
    hsv[1] -= howmuch
    result = matplotlib.colors.to_hex(matplotlib.colors.hsv_to_rgb(hsv))
    return result


def plot_csvs_relative(files: Dict[str, typing.Union[Path, Iterable[Path]]],
                       baseline: typing.Union[Path, Iterable[Path]], *,
                       label: str = "Relative overhead compared to baseline",
                       metrics: List[str] = None, metric_mapping: Callable[[str, str], str] = _default_metric_mapping,
                       progname_mapping: Callable[[str], str] = _default_progname_mapping,
                       legend_inside=True, include_variant_in_legend=False, text_in_bar=True, tick_angle: float = None,
                       customize_bar: Callable[["BarResults", go.Bar], go.Bar] = None,
                       preprocess_data: Callable[["pd.DataFrame", str], pd.DataFrame] = None,
                       include_progname_filter: Callable[[str], bool] = None,
                       add_summary_bars=None,
                       error_bar_args: go.bar.ErrorY = None) -> Tuple[go.Figure, List[BarResults]]:
    if add_summary_bars is None:
        add_summary_bars = [SummaryBar.ArithmeticMean]
    if metrics is None:
        metrics = ["cycles", "instructions", "l2cache_misses"]
    # Add support for metric + variant mapping
    if metric_mapping is _default_metric_mapping and include_variant_in_legend:
        metric_mapping = _default_metric_mapping_with_variant

    baseline_df = _load_statcounters_csv(baseline, metrics, preprocess_data, "baseline")
    baseline_medians = baseline_df.groupby("progname").median()
    fig = go.Figure()
    bar_results = []
    all_programs = set()
    # First by metric, then by configuration (ensures that e.g. cycles are consecutive and can be compared easily)
    for metric in metrics:
        for name, csv in files.items():
            orig_df = _load_statcounters_csv(csv, metrics, preprocess_data, name)
            # Normalize by basline median
            df = orig_df.apply(_normalize_values, axis=1, args=(baseline_medians,))  # type: pd.DataFrame
            # df["variant"] = name
            grouped = df.groupby("progname")
            data = dict()
            for progname, group in grouped:
                if include_progname_filter is not None and not include_progname_filter(progname):
                    print("Skipping program", progname)
                    continue
                all_programs.add(progname)
                data[progname_mapping(progname)] = group[metric]
            result = BarResults(data, metric_mapping(metric, name), raw_metric=metric)
            bar_results.append(result)
            fig.add_trace(
                result.create_bar(text_in_bar=text_in_bar, error_bar_args=error_bar_args, customize_bar=customize_bar,
                                  add_summary_bars=add_summary_bars))

    yaxis = go.layout.YAxis(title=dict(text=label, font=dict(size=18)))
    yaxis.tickformat = ',.0%'  # percentage with 0 fractional digits

    # Add alternating shading:
    shapes = []  # type: List[go.layout.Shape]
    num_bars = len(all_programs) + len(add_summary_bars)
    for i in range(0, num_bars):
        if (i % 2) == 1:
            # insert a shaded background for this bar
            s = go.layout.Shape(
                type="rect",
                # x-reference is assigned to the x-values
                # xref="x",
                xref="paper",  # actually it works better if we assign to the 0,1 range
                # y-reference is assigned to the plot paper [0,1]
                yref="paper",
                x0=i / num_bars,
                y0=0,
                x1=(i + 1) / num_bars,
                y1=1,
                fillcolor="Gainsboro",
                opacity=0.5,
                layer="below",
                line_width=0,
            )
            print("Adding shaded background from", s.x0, "to", s.x1)
            shapes.append(s)

    fig.update_layout(
        barmode='group',
        yaxis=yaxis, shapes=shapes,
        showlegend=True,
    )
    fig.update_yaxes(automargin=True, hoverformat=".2%")
    fig.update_xaxes(automargin=True)
    if tick_angle:
        fig.update_xaxes(tickangle=tick_angle)
    fig.update_layout(
        margin=dict(
            r=0,
            b=0,
            t=0,
        ),
    )
    if legend_inside:
        fig.update_layout(
            legend=go.layout.Legend(
                x=0,
                y=1,
                traceorder="normal",
                #     font=dict(
                #         family="sans-serif",
                #         size=12,
                #         color="black"
                #     ),
                bgcolor="rgba(255, 255, 255, 0.5)",
                bordercolor="Black",
                borderwidth=2
            )
        )

    return fig, bar_results


def _first_upper(s: str):
    return s[:1].upper() + s[1:]


# interesting["faster"] = latex_df[latex_df['cycles'] < 1.0]
def _latex_define_macro(prefix, name, value):
    assert isinstance(value, str)
    fullname = _first_upper(prefix) + _first_upper(name)  # camelcase
    chars = []
    # Try to remove all chars that aren't valid in a latex macro name:
    next_upper = False
    for c in fullname:
        if c in ("-", " ", "\t", "_", "."):
            next_upper = True
            continue
        if c.isalpha():
            if next_upper:
                chars.append(c.upper())
                next_upper = False
            else:
                chars.append(c)
    fullname = ''.join(chars)
    print(fullname, "=", value)
    return "\\newcommand*{\\" + str(fullname) + "}{" + str(value) + "}\n"


def _latex_bench_overhead_macro(prefix, metric: str, name: str, value: float):
    assert isinstance(value, float)
    if value < 0.0:
        suffix = "Faster"
        value = "{:.1f}\\%".format(-(value * 100.0))
    else:
        suffix = "Overhead"
        value = "{:.1f}\\%".format(value * 100.0)
    return _latex_define_macro(prefix, _first_upper(name) + _first_upper(metric) + suffix, value)


def generate_latex_macros(f: Union[typing.IO, Path], data: List[BarResults], prefix: str,
                          metrics: List[str] = None) -> None:
    if isinstance(f, Path):
        with f.open("w") as opened:
            return generate_latex_macros(opened, data, prefix, metrics)

    if metrics is None:
        metrics = ["cycles"]

    for results in data:
        if results.raw_metric not in metrics:
            continue
        metric = results.human_metric
        f.write("% START " + results.human_metric + "\n")
        # First write the overall values:
        medians = results.medians  # type: Iterable[float]
        f.write(_latex_bench_overhead_macro(prefix, metric, "Min", np.min(results.medians)))
        f.write(_latex_bench_overhead_macro(prefix, metric, "Median", np.median(results.medians)))
        f.write(_latex_bench_overhead_macro(prefix, metric, "Mean", np.mean(results.medians)))
        # For geomean we need values > 1:
        f.write(_latex_bench_overhead_macro(prefix, metric, "Geomean", gmean([x + 1 for x in results.medians]) - 1))

        # For the worst case also write the benchmark name:
        worst = max(results.benchmark_data, key=operator.attrgetter("median"))
        assert worst.median == np.max(results.medians)
        f.write(_latex_bench_overhead_macro(prefix, metric, "Max", np.max(results.medians)))
        f.write(_latex_define_macro(prefix, "Max" + _first_upper(metric) + "Benchmark", str(worst.program)))
        f.write("% Per-benchmark " + results.human_metric + "\n")
        for b in results.benchmark_data:
            f.write(_latex_bench_overhead_macro(prefix, metric, b.program.lower(), b.median))
        f.write("% END " + results.human_metric + "\n\n")
