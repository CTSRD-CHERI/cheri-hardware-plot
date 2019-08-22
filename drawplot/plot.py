#! /usr/bin/env python3
#-
# Copyright (c) 2017-2018 Alexandre Joannou
# All rights reserved.
#
# This software was developed by SRI International and the University of
# Cambridge Computer Laboratory under DARPA/AFRL contract FA8750-10-C-0237
# ("CTSRD"), as part of the DARPA CRASH research programme.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE AUTHOR AND CONTRIBUTORS ``AS IS'' AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED.  IN NO EVENT SHALL THE AUTHOR OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
# OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
# HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
# OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
# SUCH DAMAGE.
#

import numpy as np
import pandas as pd
from scipy.stats.mstats import gmean
import pprint
import matplotlib.patches as mpatches
from enum import Enum
from drawplot.data import index_cat_data, std_norm, overheads2median
from drawplot.plot_params import element_spaces_conf, element_color, element_hatch, bench_name, arch_name, tstruct_name, metric_name

from collections import defaultdict
from matplotlib import gridspec
from matplotlib.figure import Figure

# type of graph enum
GraphType = Enum('GraphType', ['bar','box'])

###############################################################################
# internal graph draw function
def draw (
        ax, families, confs, metrics, widths, tabulate,
        gtype=GraphType.bar, baseline=None, y_as_percent=False, add_average=False):
    """
    Wraps the matplotlib call to the bar() / boxplot() method.

    Positional arguments
    --------------------
    ax       -- The maplotlib Axes object to be used to render the graph.
    families -- A sequence of sequences of benchmark names.
    confs    -- A sequence of configuration tuples
             (bitfile-cpu, sdk-cpu, target-arch-cpu, table-struct).
    widths   -- An instance of then Widths namedtuple. See plot_params.py.
    tabulate -- Sequence of configs to tabulate

    Keyword arguments
    -----------------
    gtype        -- An instance of the GraphType Enum.
    baseline     -- The baseline configuration tuple.
    y_as_percent -- A boolean determining whether to display the y axis
                 as percents.

    Return
    ------
    A tuple of the Axes object, the sequence of x-ticks positions, the
    sequence of x-ticks labels and the sequence of rows to tabulate.
    """
    pos         = widths.buffers
    max_y       = None
    min_y       = None
    xticks      = []
    xticklabels = []
    rows = defaultdict(list)
    # for each family of benchmarks
    worst_and_best_case = {}
    ###########################################################################
    for i,family in enumerate(families):
        # for each benchmark in the current family
        #######################################################################
        for j,(bench,g) in enumerate(family.groupby(level='progname')):
            # init baseline if necessary
            if baseline:
                baseline_data = g.xs(
                        tuple(baseline),
                        level=(
                            'bitfile-cpu','sdk-cpu',
                            'target-arch-cpu','table-struct'))
            pos_start = pos
            # for each conf-metric pair
            ###################################################################
            for k,(conf, metric) in enumerate([(x,y)
                                                for y in metrics
                                                for x in confs]):
                conf_data = g.xs(
                        tuple(conf),
                        level=(
                            'bitfile-cpu','sdk-cpu',
                            'target-arch-cpu','table-struct'))
                row=conf+[metric]
                ############
                # Bar plot #
                ###############################################################
                if gtype == GraphType.bar:
                    if baseline:
                        y, err_low, err_high = std_norm(baseline_data[metric], conf_data[metric])
                    else:
                        # IQR or stddev?
                        if False:
                            y = conf_data[metric].mean()
                            err = conf_data[metric].std()
                            err_low = err - y
                            err_high = err - y
                        else:
                            y = conf_data[metric].median()
                            q75 = np.percentile(conf_data[metric], 75)
                            q25 = np.percentile(conf_data[metric], 25)
                            assert q25 <= y
                            assert q75 >= y
                            err_low = y - q25
                            err_high = q75 - y
                    # matplotlib expects low and high errors to be positive values! I.e. err_low is subtracted from the
                    # the value and err_high is added!
                    assert err_low >= 0, err_low
                    assert err_high >= 0, err_high
                    # print(y, err_low, err_high)
                    high_value = y + err_high
                    low_value = y - err_low
                    max_y = high_value if max_y is None else max(high_value, max_y)
                    min_y = low_value if min_y is None else min(low_value, min_y)
                    conf_str = " ".join(conf)
                    print(bench, conf_str, metric, y, "low=" + str(low_value), "high=" + str(high_value))
                    # TODO: use pandas?
                    if worst_and_best_case.get(conf_str, None) is None:
                        worst_and_best_case[conf_str] = {}
                    if worst_and_best_case[conf_str].get(metric, None) is None:
                        worst_and_best_case[conf_str][metric] = {
                            "high": {"benchmark": bench, "value": y},
                            "low": {"benchmark": bench, "value": y},
                            "values": [y],
                        }
                    else:
                        cur_max = worst_and_best_case[conf_str][metric]
                        cur_max["values"].append(y)
                        if cur_max["high"]["value"] < y:
                            cur_max["high"] = {"benchmark": bench, "value": y}
                        if cur_max["low"]["value"] > y:
                            cur_max["low"] = {"benchmark": bench, "value": y}

                    #pos += widths.elements/2.0
                    # matplotlib expects [[err_low1, err_low2], [err_hi1, err_hi2]]
                    ax.bar(pos, y, yerr=[[err_low], [err_high]],
                            width=widths.elements,
                            edgecolor=element_color(conf,k),
                            hatch=element_hatch(conf,k),
                            zorder=10,
                            color='none',
                            error_kw={
                                'ecolor': 'black',
                                'elinewidth': 1,
                                'capsize': widths.elements,
                                'zorder': 11
                            })
                    #pos += widths.elements/2.0
                    pos += widths.elements
                ###############################################################
                ############
                # Box plot #
                ###############################################################
                elif gtype == GraphType.box:
                    y = overheads2median(
                            conf_data[metric],
                            np.median(baseline_data[metric]))
                    max_y = max([max_y]+y) if max_y else max(y)
                    min_y = min([min_y]+y) if min_y else min(y)
                    pos += widths.elements/2.0
                    ax.boxplot([y],
                            positions=[pos],
                            widths=widths.elements)
                    pos += widths.elements/2.0
                ###############################################################
                if tabulate and row in tabulate:
                    if y_as_percent:
                        rows[tuple(row)].append("{:5.2f}".format(100*y if not baseline else 100*(y-1)))
                    else:
                        rows[tuple(row)].append("{:7.3f}".format(y))
                # skip space between plot elements
                if k < (len(confs) * len(metrics)) - 1:
                    pos += widths.element_spaces
            ###################################################################
            xticks.append(pos - ((pos-pos_start)/2))

            # Need to escape underscores for latex output!
            # FIXME: why doesn't matplotlib escape this??
            xticklabels.append(bench_name(bench).replace("_", "\\_"))
            # skip space between benchmarks
            if j < family.index.get_level_values('progname').unique().size - 1:
                pos += widths.bench_spaces
        #######################################################################
        # skip space between benchmark families
        if i < len(families) - 1:
            #ax.axvline(x=pos+widths.family_spaces/2.0, ymin=-100, color='grey', linewidth=1, linestyle=":", zorder=20)
            pos += widths.family_spaces
            if tabulate:
                for row in rows.values():
                    row.append('')

    print("Worst and best cases:")
    for wabc_conf in worst_and_best_case:
        print("Configuration:", wabc_conf)
        by_conf = worst_and_best_case[wabc_conf]
        for wabc_metric in by_conf:
            print("  Metric:", wabc_metric)
            by_metric = by_conf[wabc_metric]
            assert isinstance(by_metric, dict)
            values = by_metric["values"]
            assert isinstance(values, list)
            by_metric["geomean"] = gmean(values)
            by_metric["average"] = np.average(values)
            by_metric["median"] = np.median(values)
            for k, v in by_metric.items():
                print("    ", k, ": ", v, sep="")

    # pprint.pprint(worst_and_best_case)
    print("--------")
    ###########################################################################
    # y axis handling
    if gtype == GraphType.bar:
        nb_decimals=2
        nb_steps = 7
        if baseline:
            ax.axhline(y=1,color='red', linewidth=1, linestyle="--", zorder=20)
            nb_decimals=1
            nb_steps = 6
        step = np.round((max_y - min_y) / nb_steps,decimals=nb_decimals)
        while step == 0:
            nb_decimals += 1
            step = np.round((max_y - min_y) / nb_steps,decimals=nb_decimals)
        yhi = max_y + (step - max_y % step)
        ylo = min_y - (min_y % step) if baseline else 0.0
        yticks = np.arange(np.round(ylo,decimals=nb_decimals),yhi+step,step)
        if baseline:
            yticks = np.concatenate((np.arange(1.0,ylo-step,-step)[::-1][:-1],np.arange(1.0,yhi+step,step)))
            if yticks[0] == 1.0:
                yticks = np.insert(yticks,0,1.0-step)
        # print("yticks =", yticks)
        ax.set_ylim(yticks[0], yticks[-1])
        ax.set_yticks(yticks)
        if y_as_percent:
            if baseline:
                yticks = map(lambda x: x-1.0, yticks)
            ax.set_yticklabels(map(lambda x: "{:+5.0f}\%".format(100*x),yticks))
    elif gtype == GraphType.box:
        if y_as_percent:
            ax.set_yticklabels(map(lambda x: "{:+5.0f}\%".format(100*x),ax.get_yticks()))

    # return the axis and x-ticks
    return ax, xticks, xticklabels, rows

def draw_table(
        ax, rows, benchs, confs, metrics, widths, height_scaling,
        archs_in_rowlabel, y_as_percent, sdks_in_rowlabel, bitfiles_in_rowlabel, tstructs_in_rowlabel):
    colw = [widths.buffers - widths.bench_spaces/2.0]
    cellc = ['lightgrey']
    nconfs = len(confs)
    nmetrics = len(metrics)
    for i,bs in enumerate(benchs):
        for j,b in enumerate(bs):
            colw.append(nconfs*nmetrics*widths.elements + (nconfs*nmetrics-1)*widths.element_spaces + widths.bench_spaces)
            cellc.append('white')
        if i < len(benchs) - 1:
            colw.append(widths.family_spaces - widths.bench_spaces)
            cellc.append('lightgrey')
    colw.append(widths.buffers - widths.bench_spaces/2.0)
    cellc.append('lightgrey')

    rowlbl = []
    for row in rows:
        rowlbl.append('')

    if archs_in_rowlabel:
        for k,row in enumerate(rows.keys()):
            rowlbl[k] = '{}{}'.format(arch_name(row[2]), ' (\%)' if y_as_percent else '')
            if sdks_in_rowlabel:
                rowlbl[k] = '{} ({} SDK)'.format(rowlbl[k], tstruct_name(row[1]))
            if bitfiles_in_rowlabel:
                rowlbl[k] = '{} (on {} bitfile)'.format(rowlbl[k], tstruct_name(row[0]))
            if tstructs_in_rowlabel:
                rowlbl[k] = '{} tag table\n{}'.format(tstruct_name(row[3]), rowlbl[k])
    else:
        rowlabel = None

    tbl = ax.table(
            loc='center',
            colWidths=list(map(lambda x: x/100.0,colw)),
            cellColours=[cellc]*len(rows),
            cellText=[['']+r+[''] for r in rows.values()],
            rowLabels=rowlbl)

    tbl.auto_set_font_size(False)
    tbl.set_fontsize('medium')

    for cell in tbl.properties()['child_artists']:
        dflt_h = cell.get_height()
        #print(dflt_h)
        cell.set_height(dflt_h*height_scaling)
    return ax



###############################################################################
# exported plotting function
def plot (
        fig: Figure, df: pd.DataFrame, gtype: GraphType, baseline=None, configs=None, benchs=None,
        metrics=['cycles'], tabulate=None, lbl=None, metrics_in_legend=False,
        archs_in_legend=False, sdks_in_legend=False, bitfiles_in_legend=False,
        tstructs_in_legend=False, legend_columns=None,
        archs_in_rowlabel=False, sdks_in_rowlabel=False,
        bitfiles_in_rowlabel=False, tstructs_in_rowlabel=False,
        legend_location="top-left", ylim=None, y_as_percent=False,
        size_ratios=(0.3, 0.7, 1.7, 1.2)):
    """
    Generates a graph of the specified data.

    Positional arguments
    --------------------
    fig   -- The maplotlib Figure object to be used to render the graph.
    df    -- The pandas DataFrame with the data to plot.
    gtype -- The desired graph type. An instance of the GraphType Enum.

    Keyword arguments
    -----------------
    baseline           -- The baseline configuration tuple.
    configs            -- A sequence of configuration tuples
                       (bitfile-cpu, sdk-cpu, target-arch-cpu, table-struct).
    benchs             -- A sequence of sequences of benchmark names.
    metrics            -- A sequence of metric names to plot.
    tabulate           -- A sequence of configs/metrics to tabulate.
    lbl                -- A label to display in the legend.
    metrics_in_legend  -- A boolean for displaying metric names in legend.
    archs_in_legend    -- A boolean for displaying target architecture names
                       in legend.
    sdks_in_legend     -- A boolean for displaying SDK names in legend.
    bitfiles_in_legend -- A boolean for displaying bitfile names in legend.
    tstruct_in_legend  -- A boolean for displaying tag-table structures in
                       legend.
    archs_in_rowlabel  --
    sdks_in_rowlabel   --
    bitfiles_in_rowlabel --
    tstructs_in_rowlabel --
    legend_columns     -- Number of columns in legend.
    legend_location    -- The legend location.
    ylim               -- Explicit limits on the y-axis (tuple (ymin, ymax)).
    y_as_percent       -- A boolean determining whether to display the y axis
                       as percents.
    size_ratios        -- Tuple of ratios to the element's widths
                       (element_spaces, bench_spaces, family_spaces, buffers).
    Return
    ------
    The Axes object.
    """
    # prepare params
    if configs == None:
        configs=[['cheri256','cheri256','cheri256','0']]
    prognames = df.index.get_level_values('progname').unique()
    if benchs == None:
        benchs = [prognames]
    # check that all required benchmarks are present
    else:
        for b in [b for x in benchs for b in x]:
            if not b in prognames:
                print("{} is not a valid benchmark. Available benchmarks are: {}".format(b,prognames))
                exit(-1)
    # select the rows we care about
    def f(prog, bitfile, sdk, arch, tstruct, confs, benchs_family):
        def conf_eq(conf):
            return True if conf[0] == bitfile and conf[1] == sdk and conf[2] == arch and conf[3] == tstruct else False
        if prog in benchs_family:
            return any(map(conf_eq,confs))
        return False
    df = df.reset_index()
    # filter the relevant rows
    families = []
    for bench_family in benchs:
        cnfs = configs if baseline == None else [baseline]+configs
        d = df[df[['progname','bitfile-cpu','sdk-cpu','target-arch-cpu','table-struct']].apply(lambda x: f(*x, cnfs, bench_family), axis=1)]
        d = index_cat_data(d,bench_family)
        families.append(d)
    # compute bar widths
    widths = element_spaces_conf(benchs, nb_confs=len(configs), nb_metrics=len(metrics), ratios=size_ratios)

    if not tabulate:
        ax0 = fig.add_subplot(1,1,1)
    else:
        gs = gridspec.GridSpec(2,1, height_ratios=(100,15*len(tabulate)))
        #gs = gridspec.GridSpec(2,1)
        ax1 = fig.add_subplot(gs[1])
        ax0 = fig.add_subplot(gs[0])
        fig.subplots_adjust(hspace=0.05)

    ax, xticks, xticklabels, rows = draw(ax0, families, configs, metrics, widths, tabulate, gtype, baseline, y_as_percent)

    for a in fig.get_axes():
        a.set_xlim(0,100.0)
        a.set_xticks(xticks)
        a.tick_params(which='both', bottom='off', top='off',right='off')
        a.set_xticklabels([])

    if rows:
        ax = draw_table(ax1,rows,benchs,configs,metrics,widths,2.5,archs_in_rowlabel, y_as_percent, sdks_in_rowlabel,bitfiles_in_rowlabel, tstructs_in_rowlabel)
        ax.yaxis.set_visible(False)
        for s in ax.spines.values():
            s.set_visible(False)

    ax.set_xlim(0,100.0)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels, va='top', ha='center', rotation=90)
    ax.tick_params(which='both', bottom='off', top='off',right='off')
    if ylim:
        ax0.set_ylim(*ylim)
    ax0.yaxis.grid(zorder=0)
    to_legend = []
    needs_legend = False
    if lbl:
        needs_legend = True
    for i,(conf,m) in enumerate([(x,y) for y in metrics for x in configs]):
        legend_lbl = []
        if archs_in_legend:
            needs_legend = True
            legend_lbl.append(arch_name(conf[2]))
            if sdks_in_legend:
                legend_lbl.append("({} SDK)".format(sdk_name(conf[1])))
            if bitfiles_in_legend:
                legend_lbl.append("(on {} bitfile)".format(conf[0]))
            if tstructs_in_legend:
                legend_lbl.append("({} tag table)".format(tstruct_name(conf[3])))
        if metrics_in_legend:
            needs_legend = True
            # XXXAR might need to escape _ for latex output:
            legend_lbl.append(metric_name(m).replace("_", "\\_"))
        to_legend.append(mpatches.Patch(facecolor='none', hatch=element_hatch(conf,i), edgecolor=element_color(conf,i), label=" ".join(legend_lbl)))
    if needs_legend:
        if legend_columns == None:
            legend_columns = len(metrics) if metrics_in_legend else len(configs)
        if legend_location == "top-left":
            ax0.legend(title=lbl,handles=to_legend, borderaxespad=0, ncol=legend_columns, fontsize='small', loc='upper left', bbox_to_anchor=(0,1))
        elif legend_location == 'outside':
            ax0.legend(title=lbl,handles=to_legend, borderaxespad=0, ncol=legend_columns, fontsize='small', loc='lower left', mode="expand", bbox_to_anchor=(0.0,1.02,1.0,0.02))
    return ax.get_figure()
