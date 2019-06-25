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

from collections import namedtuple

def metric_name (m):
    return {
        "cycles": " cpu cycles",
        "instructions": " cpu instructions",
        "l2cache_misses": "l2cache misses",
        "l2cache_flits": "l2cache flits"
    }.get(m, m)

def bench_name (n):
    return {
        "earley-boyer": "octane-earley-boyer",
        "gbemu": "octane-gbemu",
        "pdfjs": "octane-pdfjs",
        "splay": "octane-splay",
        "treeadd 21 1 0": "olden-treeadd",
        "perimeter 10 0": "olden-perimeter",
        "mst 1024 0": "olden-mst",
        "bisort 250000 0": "olden-bisort"
    }.get(n,n)

def arch_name (n):
    return {
        "mips": "MIPS",
        "cheri128": "128-bit CHERI",
        "cheri256": "256-bit CHERI"
    }.get(n,n)

def sdk_name (n):
    return {
        "mips": "pure MIPS",
        "cheri128": "128-bit CHERI",
        "cheri256": "256-bit CHERI"
    }.get(n,n)

def tstruct_name (n):
    return {
        "0": "flat",
        "0_256": "hierarchical"
    }.get(n,n)

def element_color(conf,i):
    clr = ['blue','red','green','orange','purple']
    return {
        ("mips","mips","0"): "orange",
        ("cheri256","mips","0"): "green",
        ("cheri256","cheri256","0"): "blue",
        ("cheri256","cheri256","0_256"): (0.2,0.45,1),
        ("cheri128","cheri128","0"): "red",
        ("cheri128","cheri128","0_256"): (1,0.45,0.2)
    }.get((conf[1],conf[2],conf[3]),clr[i%5])

def element_hatch(conf,i):
    htch = ['////','\\\\\\\\','XXX','+++','---']
    return {
        ("mips","mips","0"): "---",
        ("cheri256","mips","0"): "XXX",
        ("cheri256","cheri256","0"): "//",
        ("cheri256","cheri256","0_256"): "\\\\\\\\",
        ("cheri128","cheri128","0"): "\\\\",
        ("cheri128","cheri128","0_256"): "////"
    }.get((conf[1],conf[2],conf[3]),htch[i%5])

Widths = namedtuple('Widths', ['elements', 'element_spaces', 'bench_spaces', 'family_spaces', 'buffers'])
# compute the various with between graph elements, based on the number of
# benchmarks/benchmark families/metrics/configurations and the ratios desired.
def element_spaces_conf(samples_shape, nb_confs=1, nb_metrics=1, ratios=(0.3, 0.7, 1.7, 1.2)):
    # shape must be a list of lists of samples
    elements = 0
    element_spaces = 0
    bench_spaces = 0
    for f in samples_shape:
        elements += (nb_confs * nb_metrics) * len(f)
        element_spaces += ((nb_confs * nb_metrics) - 1) * len(f)
        bench_spaces += len(f) - 1
    family_spaces = len(samples_shape) - 1
    buffers = 2

    elements_w = 100.0 / (elements+element_spaces*ratios[0]+bench_spaces*ratios[1]+family_spaces*ratios[2]+buffers*ratios[3])
    element_spaces_w = ratios[0] * elements_w
    bench_spaces_w = ratios[1] * elements_w
    family_spaces_w = ratios[2] * elements_w
    buffers_w = ratios[3] * elements_w
    return Widths(elements_w, element_spaces_w, bench_spaces_w, family_spaces_w, buffers_w)
