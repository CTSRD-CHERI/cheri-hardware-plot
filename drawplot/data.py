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

import pandas as pd
from collections import defaultdict, namedtuple
import os.path as op
import numpy as np

###############
# import data #
###############

def index_cat_data (df: pd.DataFrame,
        benchs=[
                'automotive-basicmath',
                'automotive-bitcount',
                'automotive-qsort',
                'automotive-susan',
                'consumer-jpeg',
                'network-dijkstra',
                'network-patricia',
                'office-stringsearch',
                'security-blowfish',
                'security-rijndael',
                'security-sha',
                'telecomm-CRC32',
                'telecomm-FFT',
                'telecomm-adpcm',
                "earley-boyer",
                "gbemu",
                "pdfjs",
                "splay",
                "treeadd 21 1 0",
                "perimeter 10 0",
                "mst 1024 0",
                'bisort 250000 0',
                "pgbench",
                "scp"
            ]):
    df.set_index(['bitfile-cpu', 'sdk-cpu', 'target-arch-cpu', 'table-struct', 'progname'], inplace=True)
    return df
    """
    Takes a Pandas DataFrame of CHERI statcounters and index it with appropriate
    categorical data for bitfile-cpu, sdk-cpu, target-arch-cpu and progname.

    Keyword arguments:
    benchs -- an ordered list of names to use for the progname category
    """
    cats = {
        'bitfile-cpu': ['beri','cheri128','cheri256'],
        'sdk-cpu': ['mips','cheri128','cheri256'],
        'target-arch-cpu': ['mips','cheri128','cheri256'],
        'table-struct': ['0','0_256'],
        'progname': benchs
    }
    for idx,idx_values in cats.items():
        #df.loc[:,idx] = df.loc[:,idx].astype("category", categories=idx_values, ordered=True)
        #df.loc[:,idx] = pd.Categorical(df.loc[:,idx], categories=idx_values, ordered=True)
        assign_arg = {idx: pd.Categorical(df.loc[:,idx], categories=idx_values, ordered=True)}
        df = df.assign(**assign_arg)
    return df.set_index(list(cats.keys()))


def import_data (csv_files):
    """
    Takes a list of CHERI statcounters csv file names and opens them as a Pandas
    Dataframe. Appends to the existing column some derived metrics. Orders the
    index via a call to index_cat_data before returning the DataFrame.
    """
    # csv data
    dfs = []
    for f in csv_files:
        dfs.append(pd.read_csv(f))
    df = pd.concat(dfs)
    # compute useful higher level data
    df = df.assign(cpi = lambda x: x.cycles / x.instructions)

    df = df.assign(tlb_miss = lambda x: x.itlb_miss + x.dtlb_miss)
    df = df.assign(tlb_inst_share = lambda x: x.tlb_miss * 50 / x.instructions)

    df = df.assign(cap_bytes_read = df.apply(lambda x: 16*x['mipsmem_cap_read'] if x['bitfile-cpu'] == 'cheri128' else 32*x['mipsmem_cap_read'] if x['bitfile-cpu'] == 'cheri256' else 0, axis=1))
    df = df.assign(mem_bytes_read = lambda x: x.mipsmem_byte_read + 2*x.mipsmem_hword_read + 4*x.mipsmem_word_read + 8*x.mipsmem_dword_read + x.cap_bytes_read)
    df = df.assign(cap_bytes_write = df.apply(lambda x: 16*x['mipsmem_cap_write'] if x['bitfile-cpu'] == 'cheri128' else 32*x['mipsmem_cap_write'] if x['bitfile-cpu'] == 'cheri256' else 0, axis=1))
    df = df.assign(mem_bytes_write = lambda x: x.mipsmem_byte_write + 2*x.mipsmem_hword_write + 4*x.mipsmem_word_write + 8*x.mipsmem_dword_write + x.cap_bytes_write)
    df = df.assign(mem_bytes = lambda x: x.mem_bytes_read + x.mem_bytes_write)

    df = df.assign(icache_misses = lambda x: x.icache_read_miss + x.icache_write_miss)
    df = df.assign(icache_hits = lambda x: x.icache_read_hit + x.icache_write_hit)
    df = df.assign(icache_accesses = lambda x: x.icache_misses + x.icache_hits)
    df = df.assign(icache_read_miss_rate = lambda x: x.icache_read_miss/(x.icache_read_hit+x.icache_read_miss))
    df = df.assign(icache_read_hit_rate = lambda x: x.icache_read_hit/(x.icache_read_hit+x.icache_read_miss))

    df = df.assign(dcache_misses = lambda x: x.dcache_read_miss + x.dcache_write_miss)
    df = df.assign(dcache_hits = lambda x: x.dcache_read_hit + x.dcache_write_hit)
    df = df.assign(dcache_accesses = lambda x: x.dcache_misses + x.dcache_hits)
    df = df.assign(dcache_read_miss_rate = lambda x: x.dcache_read_miss/(x.dcache_read_hit+x.dcache_read_miss))
    df = df.assign(dcache_read_hit_rate = lambda x: x.dcache_read_hit/(x.dcache_read_hit+x.dcache_read_miss))

    df = df.assign(l2cache_misses = lambda x: x.l2cache_read_miss + x.l2cache_write_miss)
    df = df.assign(l2cache_hits = lambda x: x.l2cache_read_hit + x.l2cache_write_hit)
    df = df.assign(l2cache_accesses = lambda x: x.l2cache_misses + x.l2cache_hits)
    df = df.assign(l2cache_read_miss_rate = lambda x: x.l2cache_read_miss/(x.l2cache_read_hit+x.l2cache_read_miss))
    df = df.assign(l2cache_read_hit_rate = lambda x: x.l2cache_read_hit/(x.l2cache_read_hit+x.l2cache_read_miss))

    df = df.assign(l2cache_req_flits = lambda x: x.l2cachemaster_read_req + x.l2cachemaster_write_req_flit)
    df = df.assign(l2cache_rsp_flits = lambda x: x.l2cachemaster_read_rsp_flit + x.l2cachemaster_write_rsp)
    df = df.assign(l2cache_flits = lambda x: x.l2cache_req_flits + x.l2cache_rsp_flits)

    df = df.assign(tagcache_req_flits = lambda x: x.tagcachemaster_read_req + x.tagcachemaster_write_req_flit)
    df = df.assign(tagcache_rsp_flits = lambda x: x.tagcachemaster_read_rsp_flit + x.tagcachemaster_write_rsp)
    df = df.assign(tagcache_flits = lambda x: x.tagcache_req_flits + x.tagcache_rsp_flits)

    df = df.assign(dram_req_flits = df[['l2cache_req_flits','tagcache_req_flits']].max(axis='columns'))
    df = df.assign(dram_rsp_flits = df[['l2cache_rsp_flits','tagcache_rsp_flits']].max(axis='columns'))
    df = df.assign(dram_flits = lambda x: x.dram_req_flits + x.dram_rsp_flits)

    df = df.assign(tags_req_flits = lambda x: x.dram_req_flits - x.l2cache_req_flits)
    df = df.assign(tags_rsp_flits = lambda x: x.dram_rsp_flits - x.l2cache_rsp_flits)
    df = df.assign(tags_flits = lambda x: x.tags_req_flits + x.tags_rsp_flits)

    df = df.assign(tags_dram_overhead = lambda x: x.tags_flits / x.dram_flits)
    df = df.assign(dram_mpki = lambda x: x.dram_req_flits / (x.instructions/1000))
    df = df.assign(tags_dram_mpki = lambda x: x.tags_req_flits / (x.instructions/1000))
    df = df.assign(dram_inst_share = lambda x: x.dram_flits / x.instructions)

    return index_cat_data(df)

def no_outliers(samples,acceptable_zscore=1.75):
    zscore = np.abs((samples-samples.mean())/samples.std())
    return samples[zscore < acceptable_zscore]

def std_norm(base_samples,test_samples):
    Ea = np.mean(base_samples)
    Eb = np.mean(test_samples)
    cov_mat = np.cov(np.array([base_samples,test_samples]),ddof=1)
    coV = cov_mat[0,1]
    Va = cov_mat[0,0]
    Vb = cov_mat[1,1]
    E = Eb/Ea - coV/(Ea**2) + Eb*Va/(Ea**3)
    V = Vb/(Ea**2) - 2*Eb*coV/(Ea**3) + (Eb**2)*Va/(Ea**4)
    return (E,np.sqrt(V))

def overheads2median(samples, baseline_median):
    return list(map(lambda x: (x/baseline_median)-1.0, samples))
