#! /usr/bin/env python3
#-
# Copyright (c) 2017 Alexandre Joannou
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
import matplotlib.patches as mpatches
from enum import Enum
from drawplot.data import index_cat_data, std_norm, overheads2median
from drawplot.plot_params import element_spaces_conf, element_color, element_hatch, bench_name, arch_name, tstruct_name

# type of graph enum
GraphType = Enum('GraphType', ['bar','box'])

###############################################################################
# internal graph draw function
def draw (
        ax, families, confs, metrics, widths,
        gtype=GraphType.bar, baseline=None, y_as_percent=False):
    """
    Wraps the matplotlib call to the bar() / boxplot() method.

    Positional arguments
    --------------------
    ax       -- The maplotlib Axes object to be used to render the graph.
    families -- A sequence of sequences of benchmark names.
    confs    -- A sequence of configuration tuples
             (bitfile-cpu, sdk-cpu, target-arch-cpu, table-struct).
    widths   -- An instance of then Widths namedtuple. See plot_params.py.

    Keyword arguments
    -----------------
    gtype        -- An instance of the GraphType Enum.
    baseline     -- The baseline configuration tuple.
    y_as_percent -- A boolean determining whether to display the y axis
                 as percents.

    Return
    ------
    A tuple of the Axes object, the sequence of x-ticks positions and the
    sequence of x-ticks labels.
    """
    pos         = widths.buffers
    max_y       = None
    min_y       = None
    xticks      = []
    xticklabels = []
    # for each family of benchmarks
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
                                                for x in confs
                                                for y in metrics]):
                conf_data = g.xs(
                        tuple(conf),
                        level=(
                            'bitfile-cpu','sdk-cpu',
                            'target-arch-cpu','table-struct'))
                ############
                # Bar plot #
                ###############################################################
                if gtype == GraphType.bar:
                    y = conf_data[metric].mean()
                    err = conf_data[metric].std()
                    if baseline:
                        y,err = std_norm(
                                baseline_data[metric],
                                conf_data[metric])
                    max_y = y + err if max_y == None else max (y + err,max_y)
                    min_y = y + err if min_y == None else min (y - err,min_y)
                    #pos += widths.elements/2.0
                    ax.bar(pos, y, yerr=err,
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
                # skip space between plot elements
                if k < (len(confs) * len(metrics)) - 1:
                    pos += widths.element_spaces
            ###################################################################
            xticks.append(pos - ((pos-pos_start)/2))
            xticklabels.append(bench_name(bench))
            # skip space between benchmarks
            if j < family.index.get_level_values('progname').unique().size - 1:
                pos += widths.bench_spaces
        #######################################################################
        # skip space between benchmark families
        if i < len(families) - 1:
            #ax.axvline(x=pos+widths.family_spaces/2.0, ymin=-100, color='grey', linewidth=1, linestyle=":", zorder=20)
            pos += widths.family_spaces

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
        yhi = max_y + (step - max_y % step)
        ylo = min_y - min_y % step if baseline else 0.0
        yticks = np.arange(np.round(ylo,decimals=nb_decimals),yhi+step,step)
        if baseline:
            yticks = np.concatenate((np.arange(1.0,ylo,-step)[::-1][:-1],np.arange(1.0,yhi+step,step)))
            if yticks[0] == 1.0:
                yticks = np.insert(yticks,0,1.0-step)
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
    return ax, xticks, xticklabels

###############################################################################
# exported plotting function
def plot (
        ax, df, gtype, baseline=None, configs=None, benchs=None,
        metrics=['cycles'], lbl=None, metrics_in_legend=False,
        archs_in_legend=False, sdks_in_legend=False, bitfiles_in_legend=False,
        tstructs_in_legend=False, legend_columns=None,
        legend_location="top-left", ylim=None, y_as_percent=False,
        size_ratios=(0.3, 0.7, 1.7, 1.2)):
    """
    Generates a graph of the specified data.

    Positional arguments
    --------------------
    ax    -- The maplotlib Axes object to be used to render the graph.
    df    -- The pandas DataFrame with the data to plot.
    gtype -- The desired graph type. An instance of the GraphType Enum.

    Keyword arguments
    -----------------
    baseline           -- The baseline configuration tuple.
    configs            -- A sequence of configuration tuples
                       (bitfile-cpu, sdk-cpu, target-arch-cpu, table-struct).
    benchs             -- A sequence of sequences of benchmark names.
    metrics            -- A sequence of metric names to plot.
    lbl                -- A label to display in the legend.
    metrics_in_legend  -- A boolean for displaying metric names in legend.
    archs_in_legend    -- A boolean for displaying target architecture names
                       in legend.
    sdks_in_legend     -- A boolean for displaying SDK names in legend.
    bitfiles_in_legend -- A boolean for displaying bitfile names in legend.
    tstruct_in_legend  -- A boolean for displaying tag-table structures in
                       legend.
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
    if benchs == None:
        benchs = [df.index.get_level_values('progname').unique()]
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

    ax, xticks, xticklabels = draw(ax, families, configs, metrics, widths, gtype, baseline, y_as_percent)

    ax.set_xlim(0,100.0)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels, va='top', ha='center', rotation=90)
    ax.tick_params(which='both', bottom='off', top='off',right='off')
    if ylim:
        ax.set_ylim(*ylim)
    ax.yaxis.grid(zorder=0)
    to_legend = []
    needs_legend = False
    if lbl:
        needs_legend = True
    for i,(conf,m) in enumerate([(x,y) for x in configs for y in metrics]):
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
            legend_lbl.append(metric_name(m))
        to_legend.append(mpatches.Patch(facecolor='none', hatch=element_hatch(conf,i), edgecolor=element_color(conf,i), label=" ".join(legend_lbl)))
    if needs_legend:
        if legend_columns == None:
            legend_columns = len(metrics) if metrics_in_legend else len(configs)
        if legend_location == "top-left":
            ax.legend(title=lbl,handles=to_legend, borderaxespad=0, ncol=legend_columns, fontsize='small', loc='upper left', bbox_to_anchor=(0,1))
        elif legend_location == 'outside':
            ax.legend(title=lbl,handles=to_legend, borderaxespad=0, ncol=legend_columns, fontsize='small', loc='lower left', mode="expand", bbox_to_anchor=(0.0,1.02,1.0,0.02))
    return ax
