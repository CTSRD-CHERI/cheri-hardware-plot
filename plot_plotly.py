from pathlib import Path
from typing import *
import plotly.graph_objects as go
import pandas as pd
import numpy as np
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


def _load_statcounters_csv(csv: Path, metrics: List[str]) -> pd.DataFrame:
    df = pd.read_csv(csv)
    # add l2cache_misses stat
    df = df.assign(l2cache_misses=lambda x: x.l2cache_read_miss + x.l2cache_write_miss)
    df = df[["progname"] + metrics]
    return df


def _default_metric_mapping(m: str):
    return {
        "cycles": " CPU cycles",
        "instructions": " Instructions",
        "inst_user": " Instructions (userspace)",
        "inst_kernel": " Instructions (kernel)",
        "l2cache_misses": "L2-cache misses",
        "l2cache_flits": "L2-cache flits",
    }.get(m, m)


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

    def error_bars(self):
        return dict(type='data', symmetric=False, array=self.p75s - self.medians,
                    arrayminus=self.medians - self.p25s)

    def create_bar(self):
        return go.Bar(
            name=self.metric,
            x=[x.program for x in self.benchmark_data],
            y=self.medians,
            error_y=self.error_bars(),
            text=["{0:.2f}%".format(x * 100) for x in self.medians],
            textfont=dict(size=18),
            textposition='auto',
        )


def plot_csvs_relative(files: Dict[str, Path], baseline: Path, *, label: str = "Relative overhead compared to baseline",
                       metrics: List[str]=None, metric_mapping: Callable[[str], str] = _default_metric_mapping,
                       progname_mapping: Callable[[str], str] = _default_progname_mapping,
                       legend_inside=True) -> Tuple[go.Figure, List[BarResults]]:
    if metrics is None:
        metrics = ["cycles", "instructions", "l2cache_misses"]
    baseline_df = _load_statcounters_csv(baseline, metrics)
    baseline_medians = baseline_df.groupby("progname").median()
    fig = go.Figure()
    bar_results = []
    for name, csv in files.items():
        orig_df = _load_statcounters_csv(csv, metrics)
        # Normalize by basline median
        df = orig_df.apply(_normalize_values, axis=1, args=(baseline_medians,))  # type: pd.DataFrame
        # df["variant"] = name
        grouped = df.groupby("progname")
        for metric in metrics:
            data = dict()
            for progname, group in grouped:
                data[progname_mapping(progname)] = group[metric]
            result = BarResults(data, metric_mapping(metric))
            bar_results.append(result)
            fig.add_trace(result.create_bar())

    yaxis = go.layout.YAxis(title=dict(text=label, font=dict(size=18)))
    yaxis.tickformat = ',.0%'  # percentage with 0 fractional digits
    fig.update_layout(
        barmode='group',
        yaxis=yaxis,
        showlegend=True,
    )
    fig.update_yaxes(automargin=True)
    fig.update_xaxes(automargin=True)
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
