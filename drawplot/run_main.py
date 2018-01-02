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

from drawplot.data import import_data
from drawplot.plot import plot, GraphType
#from drawplot.box_plot import box_plot

import os.path as op
import matplotlib.pyplot as plt
from matplotlib import rc
rc('font',**{'family':'serif','serif':['Computer Modern Roman']})
rc('text', usetex=True)

#################
# main function #
#################

def main(args):

    data = import_data(args.csv)
    #data.to_csv("toto.csv")

    #########
    # graph #
    #########

    ext = "pdf"
    outfile = op.join(op.curdir,"{}.{}".format(args.subcmd,ext))
    fig = plt.figure()
    if args.subcmd in ["bar-plot","bar-plot-overheads","box-plot-overheads"]:
        kwargs = {
            'configs': args.config,
            'metrics': args.metrics,
            'benchs': args.benchs,
            'lbl': args.label,
            'ylim': args.ylim,
            'size_ratios': args.size_ratios,
            'archs_in_legend': args.archs_in_legend,
            'sdks_in_legend': args.sdks_in_legend,
            'bitfiles_in_legend': args.bitfiles_in_legend,
            'tstructs_in_legend': args.tstructs_in_legend,
            'legend_columns': args.legend_columns,
            'legend_location': args.legend_location,
            'metrics_in_legend': args.metrics_in_legend,
            'y_as_percent': args.y_as_percent,
            'tabulate': args.tabulate,
            'archs_in_rowlabel': args.archs_in_rowlabel,
            'sdks_in_rowlabel': args.sdks_in_rowlabel,
            'bitfiles_in_rowlabel': args.bitfiles_in_rowlabel,
            'tstructs_in_rowlabel': args.tstructs_in_rowlabel
        }
        if args.subcmd == "bar-plot-overheads":
            plot(fig, data, gtype=GraphType.bar, baseline=args.baseline, **kwargs)
        elif args.subcmd == "bar-plot":
            plot(fig, data, gtype=GraphType.bar, **kwargs)
        elif args.subcmd == "box-plot-overheads":
            #box_plot(fig, data, baseline=args.baseline, **kwargs)
            plot(fig, data, gtype=GraphType.box, baseline=args.baseline, **kwargs)
        outfile = op.join(op.curdir,"{}-{}.{}".format(args.subcmd,"_".join(args.metrics),ext))
    else:
        exit(-1)

    if args.output:
        outfile = args.output
    fig.set_size_inches(args.fig_size)
    fig.savefig(outfile,bbox_inches = 'tight')

    #######
    # end #
    #######
    exit(0)