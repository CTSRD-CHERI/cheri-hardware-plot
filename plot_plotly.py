from pathlib import Path
from typing import *
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import typing
import functools
from pathlib import Path


# We want to use IQRs for benchmarks
# Dived all values in the relative samples by the median of the baseline
def std_norm(base_samples, test_samples):
    if len(base_samples) == 0 or len(test_samples) == 0:
        print("base:", base_samples)
        print("test:", test_samples, flush=True)
        raise RuntimeError("EMPTY SAMPLES INPUT?")
    test_norm = test_samples / np.median(base_samples)
    # It seems like we get ZERO l2cache missing in the bzip benchmark:
    if any(np.isnan(x) for x in test_norm):
        print("base:", base_samples)
        print("test:", test_samples, flush=True)
        print("base median: ", np.median(base_samples))
        print("test_norm =", test_norm, flush=True)
        raise RuntimeError("GOT NAN IN TEST_NORM: ", test_norm)
    # per https://stackoverflow.com/questions/23228244/how-do-you-find-the-iqr-in-numpy
    # iqr = np.subtract(*np.percentile(test_norm, [75, 25]))
    q75 = np.percentile(test_norm, 75)
    q25 = np.percentile(test_norm, 25)
    med = np.median(test_norm)
    return np.median(test_norm), med - q25, q75 - med


def _old(df):
    # HACK: work around "Function names must be unique, found multiple named percentile " error
    # Seems like functools.partial won't work...
    def _p75(x):
        return np.percentile(x, 75)

    def _p25(x):
        return np.percentile(x, 25)

    # agg_dict = dict()
    # for metric in metrics:
    #     agg_dict["min_" + metric] = pd.NamedAgg(column=metric, aggfunc='min')
    #     agg_dict["max_" + metric] = pd.NamedAgg(column=metric, aggfunc='max')
    #     agg_dict["mean_" + metric] = pd.NamedAgg(column=metric, aggfunc=np.mean)
    #     agg_dict["median_" + metric] = pd.NamedAgg(column=metric, aggfunc=np.median)
    #     # agg_dict["p75_" + metric] = pd.NamedAgg(column=metric, aggfunc=functools.partial(np.percentile, q=75))
    #     # agg_dict["p25_" + metric] = pd.NamedAgg(column=metric, aggfunc=functools.partial(np.percentile, q=25))
    #     agg_dict["p75_" + metric] = pd.NamedAgg(column=metric, aggfunc=p75)
    #     agg_dict["p25_" + metric] = pd.NamedAgg(column=metric, aggfunc=p25)
    # grouped_df = norm_df.groupby("target-arch-cpu").agg(**agg_dict)  # Note: must unpack dict otherwise we get an error
    # print(grouped_df)
    # grouped_df = grouped_df.sub(1)
    # df = grouped_df
    pass


def _normalize_values(row: pd.Series, baseline_medians: pd.DataFrame):
    progname = str(row.progname)
    for k, v in row.iteritems():
        if not isinstance(v, str):
            row[k] = row[k] / baseline_medians.loc[progname][k]
    return row


def _load_statcounters_csv(csv: Union[Path, Iterable[Path]], metrics: List[str]) -> pd.DataFrame:
    if isinstance(csv, Path):
        df = pd.read_csv(csv)
    else:
        assert len(csv) > 0
        df = pd.concat(pd.read_csv(c) for c in csv)
    # add l2cache_misses stat
    df = df.assign(l2cache_misses=lambda x: x.l2cache_read_miss + x.l2cache_write_miss)
    df = df[["progname"] + metrics]
    return df


def _default_metric_mapping(m: str, variant: str):
    return {
        "cycles": " CPU cycles",
        "instructions": " Instructions",
        "inst_user": " Instructions (userspace)",
        "inst_kernel": " Instructions (kernel)",
        "l2cache_misses": "L2-cache misses",
        "l2cache_flits": "L2-cache flits",
    }.get(m, m)


def _default_metric_mapping_with_variant(m: str, variant: str):
    suffix = {
        "cycles": " CPU cycles",
        "instructions": " Instructions",
        "inst_user": " Instructions (userspace)",
        "inst_kernel": " Instructions (kernel)",
        "l2cache_misses": "L2-cache misses",
        "l2cache_flits": "L2-cache flits",
    }.get(m, m)
    return variant + " " + suffix


def _default_progname_mapping(n):
    return n


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
    def __init__(self, data: Mapping[str, pd.Series], metric: str):
        self.benchmark_data = []
        self.metric = metric
        for k, v in data.items():
            self.benchmark_data.append(_RelativeBenchmarkData(v, metric, k))
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

    def create_bar(self, text_in_bar: bool, error_bar_args: go.bar.ErrorY):
        return go.Bar(
            name=self.metric,
            x=[x.program for x in self.benchmark_data],
            y=self.medians,
            error_y=self.error_bars(error_bar_args),
            text=["{0:.2f}%".format(x * 100) for x in self.medians] if text_in_bar else None,
            # textfont=dict(size=18),
            textposition='auto',
        )


def plot_csvs_relative(files: Dict[str, typing.Union[Path, Iterable[Path]]],
                       baseline: typing.Union[Path, Iterable[Path]], *,
                       label: str = "Relative overhead compared to baseline",
                       metrics: List[str] = None, metric_mapping: Callable[[str, str], str] = _default_metric_mapping,
                       progname_mapping: Callable[[str], str] = _default_progname_mapping,
                       legend_inside=True, include_variant_in_legend=False, text_in_bar=True, tick_angle: float = None,
                       error_bar_args: go.bar.ErrorY = None) -> Tuple[go.Figure, List[BarResults]]:
    if metrics is None:
        metrics = ["cycles", "instructions", "l2cache_misses"]
    # Add support for metric + variant mapping
    if metric_mapping is _default_metric_mapping and include_variant_in_legend:
        metric_mapping = _default_metric_mapping_with_variant

    baseline_df = _load_statcounters_csv(baseline, metrics)
    baseline_medians = baseline_df.groupby("progname").median()
    fig = go.Figure()
    bar_results = []
    all_programs = set()
    for name, csv in files.items():
        orig_df = _load_statcounters_csv(csv, metrics)
        # Normalize by basline median
        df = orig_df.apply(_normalize_values, axis=1, args=(baseline_medians,))  # type: pd.DataFrame
        # df["variant"] = name
        grouped = df.groupby("progname")
        for metric in metrics:
            data = dict()
            for progname, group in grouped:
                all_programs.add(progname)
                data[progname_mapping(progname)] = group[metric]
            result = BarResults(data, metric_mapping(metric, name))
            bar_results.append(result)
            fig.add_trace(result.create_bar(text_in_bar=text_in_bar, error_bar_args=error_bar_args))

    yaxis = go.layout.YAxis(title=dict(text=label, font=dict(size=18)))
    yaxis.tickformat = ',.0%'  # percentage with 0 fractional digits

    # Add alternating shading:
    shapes = []  # type: List[go.layout.Shape]
    num_bars = len(all_programs)
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
    fig.update_yaxes(automargin=True)
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
                #     bgcolor="LightSteelBlue",
                bordercolor="Black",
                borderwidth=2
            )
        )

    return fig, bar_results
