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

import subprocess as sub
import pandas as pd

def jobargs_str (a,b,c,d):
    return "BITFILE_CPU={},SDK_CPU={},TARGET_ARCH_CPU={},TSTRUCT={},label=bluehive".format(a,b,c,d)
def artifact_str (a,b,c,d,e,f):
    return "statcounters-jenkins-{}-BITFILE_CPU={}_SDK_CPU={}_TARGET_ARCH_CPU={}_TSTRUCT={}_label=bluehive-{}.csv".format(a,b,c,d,e,f)
def conf_str (a,b,c,d):
    return "{}-{}-{}-{}".format(a,b,c,d)

# Jenkins username and password
user="readonly"
password="aiB8ax4iewithoh2"
jenkins="https://ctsrd-build.cl.cam.ac.uk"

curl = sub.run(["which","curl"],stdout=sub.PIPE).stdout.decode("utf-8").strip()
base_cmd = [curl,"--fail","-O","-k","-u","{}:{}".format(user,password)]

#duktape_jobs = [113,114,115,116,117,118,119,120,125,126,127,128,129,130]
#olden_jobs   = [271,272,273] # don't use 274
#mibench_jobs = [234,235] # don't use 236 (matrix reloaded only certain sub-jobs)
#duktape_jobs = [145,153,161,162,167,168,169,173,174,176,177,185,192,193]
#olden_jobs   = [297,298]
#mibench_jobs = [244,245]
# starting 10th august / cheriabi deadline
#duktape_jobs = [200,201,202,203,204,205,206,210,211,212]
#olden_jobs   = [321,325] # don't use 324
#mibench_jobs = [251,252,253,254,257]
# starting 19th august
duktape_jobs = [232,233,234,240,241,243,246,249,250,251,252,254,255,256,266,267]
olden_jobs   = [342,343,350,352,353,354,355,360]
mibench_jobs = [268,275,278,279,280,281,285,286]

jobs=[("bluehive-benchmark-octane-duktape", duktape_jobs)]
jobs+=[("bluehive-benchmark-olden", olden_jobs)]
jobs+=[("bluehive-benchmark-mibench", mibench_jobs)]

bitfile_cpus  = ["beri","cheri128","cheri256"]
tgt_arch_cpus = ["mips","cheri128","cheri256"]
sdk_cpus      = ["mips","cheri128","cheri256"]
tstructs      = ["0","0_256"]

#confs = [ (a,b,c,d) for a in bitfile_cpus
#                    for b in sdk_cpus
#                    for c in tgt_arch_cpus
#                    for d in tstructs
#                    if not (d == "0_256" and a == "beri")
#                    if  (a == "beri" and b == "mips" and c == "mips") or
#                        (a == "cheri128" and b != "cheri256" and c != "cheri256" and not (b == "mips" and c == "cheri128")) or
#                        (a == "cheri256" and b != "cheri128" and c != "cheri128" and not (b == "mips" and c == "cheri256"))
#                    ]
#for bitfile_cpu, sdk_cpu, tgt_arch_cpu, tstruct in confs:
#    print(conf_str(bitfile_cpu,sdk_cpu,tgt_arch_cpu,tstruct))
#
#    for job, jobnum in [(a,b) for a,a1 in jobs for b in a1]:
#        filename=artifact_str(job,bitfile_cpu,sdk_cpu,tgt_arch_cpu,tstruct,jobnum)
#        url = jenkins
#        url += "/job/"+job
#        url += "/"+jobargs_str(bitfile_cpu,sdk_cpu,tgt_arch_cpu,tstruct)
#        url += "/{}/artifact/".format(jobnum)+filename
#        run_cmd = base_cmd + [url]
#        print(" ".join(run_cmd))
#        sub.run(run_cmd)

confs = [ (a,b,c,d,e,f) for a in bitfile_cpus
                    for b in sdk_cpus
                    for c in tgt_arch_cpus
                    for d in tstructs
                    for e,e1 in jobs
                    for f in e1
                    if not (d == "0_256" and a == "beri")
                    if  (a == "beri" and b == "mips" and c == "mips") or
                        (a == "cheri128" and b != "cheri256" and c != "cheri256" and not (b == "mips" and c == "cheri128")) or
                        (a == "cheri256" and b != "cheri128" and c != "cheri128" and not (b == "mips" and c == "cheri256"))
                    ]
dfs = []
for bitfile_cpu, sdk_cpu, tgt_arch_cpu, tstruct, job, jobnum in confs:
    filename=artifact_str(job,bitfile_cpu,sdk_cpu,tgt_arch_cpu,tstruct,jobnum)
    url = jenkins
    url += "/job/"+job
    url += "/"+jobargs_str(bitfile_cpu,sdk_cpu,tgt_arch_cpu,tstruct)
    url += "/{}/artifact/".format(jobnum)+filename
    run_cmd = base_cmd + [url]
    print(" ".join(run_cmd))
    sub.run(run_cmd)
    data = pd.read_csv(filename)
    data.insert(0,'table-struct',tstruct)
    data.insert(0,'target-arch-cpu',tgt_arch_cpu)
    data.insert(0,'sdk-cpu',sdk_cpu)
    data.insert(0,'bitfile-cpu',bitfile_cpu)
    del data['archname']
    dfs.append(data)
df = pd.concat(dfs)
df.sort_values(["bitfile-cpu","sdk-cpu","target-arch-cpu","table-struct","progname"],inplace=True)
df.to_csv("hardware-results.csv",index=False)
