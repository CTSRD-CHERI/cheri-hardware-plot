#!/usr/bin/env python3
#-
# Copyright (c) 2019 Alexandre Joannou
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

import argparse as ap
import pandas as pd
import matplotlib.pyplot as plt
#import plot

################################
# Parse command line arguments #
################################################################################

# helper strings
showcat = ("Use the 'show-categories' sub-command for a list of categories "
           "available in the provided CSV(s) and the various values they can "
           "each take.")
showmet = ("Use the 'show-metrics' sub-command for a list of metrics available "
           "in the provided CSV(s).")

parser = ap.ArgumentParser(description = 'Plot csv data')

subcmds = parser.add_subparsers(
  dest = 'subcmd', metavar = 'sub-command',
  help = ("Individual sub-command help available by invoking it with "
          "-h or --help.")
)
subcmds.required = True

# common parser
common = ap.ArgumentParser(add_help=False)
common.add_argument(
  'csv', type = str, nargs = '+', metavar = 'CSV_FILE',
  help = "CSV_FILE(s) to used"
)

class BuildFilter(ap.Action):
  def __init__(self, option_strings, dest, nargs, **kwargs):
    if nargs is not 2:
      raise ValueError("nargs not allowed")
    super(BuildFilter, self).__init__(option_strings, dest, nargs, **kwargs)
  def __call__(self, parser, namespace, values, option_string=None):
    filters = getattr(namespace, self.dest)
    if filters:
      if values[0] in filters.keys():
        filters[values[0]].append(values[1])
      else:
        filters[values[0]] = [values[1]]
    else:
      filters = { values[0]: [values[1]] }
    setattr(namespace, self.dest, filters)

common.add_argument(
  '--filter-in', type = str, nargs = 2, action = BuildFilter,
  metavar = ('CATEGORY','NAME'),
  help = "Keeps ONLY the rows with a CATEGORY of NAME. If called multiple times"
         " for a same CATEGORY, keeps the rows with any of the specified NAMEs."
         "If called multiple times with different CATEGORies, keeps the rows "
         " with all specified NAMEs for each CATEGORY. " + showcat
)

#common.add_argument(
#  '--filter-out', type = str, nargs = 2, action = 'append',
#  metavar = ('CATEGORY','NAME'),
#  help = "Excludes the rows with a CATEGORY of NAME. " + showcat
#)

# bars parser
bars = ap.ArgumentParser(add_help=False)
bars.add_argument(
  '--group-by', default = None, type = str, action = 'append',
  metavar = "CATEGORY",
  help = "A CATEGORY to group by. Repeated calls will group by CATEGORies in "
         "the order of the calls. " + showcat
)
bars.add_argument(
  '--metric', default = None, type = str, action = 'append',
  help = """Metric to plot. Repeated calls allow for ploting of multiple metrics
            on the same graph. """ + showmet
)

# specific commands
subcmds.add_parser(
  'show-categories', parents = [common],
  help = "Reads the CSV_FILE(s) and shows the available categories"
)
subcmds.add_parser(
  'show-metrics', parents = [common],
  help = "Reads the CSV_FILE(s) and shows the available metrics"
)
subcmds.add_parser(
  'plot-bars', parents = [common, bars],
  help = "Generates a bar plot"
)

# helpers
################################################################################

def find_categories (df):
  df_non_num = df.select_dtypes(include='object')
  d = {}
  for key in df_non_num.keys():
    d.update({key: df_non_num[key].unique().tolist()})
  return d

def print_categories (categories):
  print("Available categories:")
  s = ""
  for k, vs in categories.items():
    s += "{:s}:\n  {:s}\n".format(k, ", ".join(vs))
  print(s)

# set categories on dataframe
def categorize_dataframe (df, categories):
  df_cat = df
  for idx, idx_values in categories.items():
    assign_arg = {idx: pd.Categorical(df_cat.loc[:,idx], categories=idx_values, ordered=True)}
    df_cat = df_cat.assign(**assign_arg)
  return df_cat.set_index(list(categories.keys()))

def find_metrics (df):
  return list(df.select_dtypes(exclude='object').keys())

def print_metrics (metrics):
  print("Available metrics:")
  print("{:s}\n".format(", ".join(metrics)))

# Main entry point
################################################################################

if __name__ == "__main__":

  # parse the arguments
  ##############################################################################
  args = parser.parse_args()

  # prepare csv metadata
  ##############################################################################
  dfs = []
  for f in args.csv:
    dfs.append(pd.read_csv(f))
  df = pd.concat(dfs)
  # identify categories
  pre_filter_categories = find_categories(df)

  # sanitize filter args
  ##############################################################################
  pre_filter_categories_keys = list(pre_filter_categories.keys())
  if args.filter_in:
    for c, ns in args.filter_in.items():
      if c not in pre_filter_categories_keys:
        print("'{:s}' is not a valid category for use with --filter-in. Please select from {:s}".format(c, str(pre_filter_categories_keys)))
        exit(-1)
      for n in ns:
        if n not in pre_filter_categories[c]:
          print("'{:s}' is not a valid name for use with --filter-in with the category '{:s}'. Please select from {:s}".format(n, c, str(pre_filter_categories[c])))
          exit(-1)
  #if args.filter_out:
  #  for c, n in args.filter_in:
  #    if c not in pre_filter_categories_keys:
  #      print("'{:s}' is not a valid category for use with --filter-out. Please select from {:s}".format(c, str(pre_filter_categories_keys)))
  #      exit(-1)
  #    if n not in pre_filter_categories[c]:
  #      print("'{:s}' is not a valid name for use with --filter-out with the category '{:s}'. Please select from {:s}".format(n, c, str(pre_filter_categories[c])))
  #      exit(-1)

  # filter dataframe and evaluate remaining categories / metrics
  ##############################################################################
  if args.filter_in:
    def build_bool_frames (c, ns):
      in_cat_frames = [df[c] == n for n in ns]
      cat_frames = in_cat_frames[0]
      for frame in in_cat_frames[1:]:
        cat_frames = cat_frames | frame
      return cat_frames
    bool_frames = [build_bool_frames(c, ns) for c, ns in args.filter_in.items()]
    samples = bool_frames[0]
    for frame in bool_frames[1:]:
      samples = samples & frame
    df = df[samples]
  # identify categories
  categories = find_categories(df)
  # identify metrics
  metrics = find_metrics(df)
  # categorize df
  df = categorize_dataframe(df, categories)

  # sanitize group/metric args
  ##############################################################################
  categories_keys = list(categories.keys())
  if hasattr(args, 'metric') and args.metric:
    for m in args.metric:
      if m not in metrics:
        print("'{:s}' is not a valid metric for use with --metric. Please select from {:s}".format(m, metrics))
        exit(-1)
  if hasattr(args, 'group_by') and args.group_by:
    for c in args.group_by:
      if c not in categories_keys:
        print("'{:s}' is not a valid category for use with --group-by. Please select from {:s}".format(c, str(categories_keys)))
        exit(-1)

  #print(df)
  #sample2 = categorize_dataframe(sample, find_categories(sample))
  #sample2.groupby('target-arch-cpu').mean()['cycles'].plot.bar()

  # identify a subcommand
  ##############################################################################
  if args.subcmd == "show-categories":
    print_categories(categories)
  elif args.subcmd == "show-metrics":
    print_metrics(metrics)
  elif args.subcmd == "plot-bars":
    #fig = plt.figure()
    print(args.metric)
    print(args.group_by)
    for m in (args.metric if args.metric else metrics):
      #df.mean(level=args.group_by)[m].plot.bar()
      #.dropna()
      print(m)
      if args.group_by:
        samples = df.mean(level=args.group_by)[m]
        for i, v in samples.iteritems():
          print("{:s}: {:f}".format(str(i), v))
        if len(args.group_by) > 1:
          ax = samples.unstack().plot.bar()
        else:
          ax = samples.plot.bar()
      else:
        ax = df[m].plot.bar()
      plt.show()
  else:
    print("No valid sub-command was found, try using one of {:s}".format(list(subcmds.choices)))
    exit(-1)
  exit(0)
