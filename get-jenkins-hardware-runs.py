#! /usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK
# -
# Copyright (c) 2017 Alexandre Joannou
# Copyright (c) 2018 Alex Richardson
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
import argparse
import json
import subprocess as sub
import pandas as pd
import sys
from collections import namedtuple


def jobargs_str(bitfile, isa, sdk, tgt_arch, tstruct):
    isa_arg = ",ISA=" + isa if isa is not None else ""
    return "BITFILE_CPU={}{},SDK_CPU={},TARGET_ARCH_CPU={},TSTRUCT={},label=bluehive".format(bitfile, isa_arg, sdk,
                                                                                             tgt_arch, tstruct)


def artifact_str(a, bitfile, isa, c, d, e, f):
    isa_arg = "_ISA=" + isa if isa is not None else ""
    return "statcounters-jenkins-{}-BITFILE_CPU={}{}_SDK_CPU={}_TARGET_ARCH_CPU={}_TSTRUCT={}_label=bluehive-{}.csv".format(
        a, bitfile, isa_arg, c, d, e, f)


def conf_str(a, b, c, d):
    return "{}-{}-{}-{}".format(a, b, c, d)


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--latest-only", action="store_true",
                    help="Only fetch the csv for the last successful run")
parser.add_argument("--subset", help="Subset of job csvs to fetch", default="19aug2017",
                    choices=["19aug2017", "cheriabi", "cap-table", "latest"])
parser.add_argument("--jobs", nargs=argparse.ONE_OR_MORE, help="Subset of job csvs to fetch",
                    choices=["olden", "mibench", "duktape"], default=["olden", "mibench", "duktape"])
try:
    import argcomplete
    argcomplete.autocomplete(parser)
except ImportError as e:
    pass
args = parser.parse_args()

# Jenkins username and password
user = "readonly"
password = "invalid"
pw_file = Path.home() / ".config" / "ctsrd-jenkins-readonly-user.txt"
try:
    password = pw_file.read_text().strip()  # remove newline
except:
    sys.exit("Could not read jenkins readonly user password from " + str(pw_file))
jenkins = "https://ctsrd-build.cl.cam.ac.uk"
LAST_SUCCESSFUL_JOB_NUM = sys.maxsize

curl = shutil.which("curl")
base_cmd = [curl, "--fail", "-O", "-k", "-u", "{}:{}".format(user, password)]

# duktape_jobs = [113,114,115,116,117,118,119,120,125,126,127,128,129,130]
# olden_jobs   = [271,272,273] # don't use 274
# mibench_jobs = [234,235] # don't use 236 (matrix reloaded only certain sub-jobs)
# duktape_jobs = [145,153,161,162,167,168,169,173,174,176,177,185,192,193]
# olden_jobs   = [297,298]
# mibench_jobs = [244,245]
if args.subset == "cheriabi":
    # starting 10th august / cheriabi deadline
    duktape_jobs = [200,201,202,203,204,205,206,210,211,212]
    olden_jobs   = [321,325] # don't use 324
    mibench_jobs = [251,252,253,254,257]
# starting 19th august
elif args.subset == "19aug2017":
    duktape_jobs = [232,233,234,240,241,243,246,249,250,251,252,254,255,256,266,267]
    olden_jobs   = [342,343,350,352,353,354,355,360]
    mibench_jobs = [268,275,278,279,280,281,285,286]
# including cap-table
elif args.subset == "cap-table":
    duktape_jobs = []
    olden_jobs = [550, 560]
    mibench_jobs = [496, 503]

# Allow fetching only the latest job
if args.subset == "latest" or args.latest_only:
    # only latest:
    duktape_jobs = [LAST_SUCCESSFUL_JOB_NUM]
    olden_jobs = [LAST_SUCCESSFUL_JOB_NUM]
    mibench_jobs = [LAST_SUCCESSFUL_JOB_NUM]


# XXXAR: no idea what the actual job number is where this was changed, let's just use something
def includes_tstruct_0(job, job_num):
    # TODO: find the right numbers
    return job_num < 400


def includes_mips_sdk(job, job_num):
    # TODO: find the right numbers
    return job_num < 400


def includes_mips_on_256_bitfile(job, job_num):
    # most runs use the cheri128 bitfile
    # TODO: do we need this?
    return job_num < 400


def includes_beri_bitfile(job, job_num):
    # most runs use the cheri128 bitfile for running MIPS (for memcpy performance)
    return job_num < 400


def includes_isa_column(job, job_num):
    if job == "bluehive-benchmark-olden":
        return job_num > 550
    if job == "bluehive-benchmark-mibench":
        return job_num > 493
    elif job == "bluehive-benchmark-octane-duktape":
        return False
    assert False, "Bad job " + job


jobs = []
if "duktape" in args.jobs:
    jobs += [("bluehive-benchmark-octane-duktape", duktape_jobs)]
if "olden" in args.jobs:
    jobs += [("bluehive-benchmark-olden", olden_jobs)]
if "mibench" in args.jobs:
    jobs += [("bluehive-benchmark-mibench", mibench_jobs)]

bitfile_cpus = ["beri", "cheri128", "cheri256"]
tgt_arch_cpus = ["mips", "cheri128", "cheri256"]
sdk_cpus = ["mips", "cheri128", "cheri256"]
tstructs = ["0", "0_256"]

# confs = [ (a,b,c,d) for a in bitfile_cpus
#                    for b in sdk_cpus
#                    for c in tgt_arch_cpus
#                    for d in tstructs
#                    if not (d == "0_256" and a == "beri")
#                    if  (a == "beri" and b == "mips" and c == "mips") or
#                        (a == "cheri128" and b != "cheri256" and c != "cheri256" and not (b == "mips" and c == "cheri128")) or
#                        (a == "cheri256" and b != "cheri128" and c != "cheri128" and not (b == "mips" and c == "cheri256"))
#                    ]
# for bitfile_cpu, sdk_cpu, tgt_arch_cpu, tstruct in confs:
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


def is_valid_job_config(bitfile, isa, sdk, target_arch, tstruct, job_name, job_num):
    # we run mips on cheri128 starting after some job
    if bitfile == "beri":
        if not includes_beri_bitfile(job_name, job_num):
            return False
        if tstruct == "0_256":
            return False
    # no more tstruct 0 for CHERI builds after some job#
    elif tstruct == "0" and not includes_tstruct_0(job_name, job_num):
        return False
    # no more MIPS sdk builds after some build:
    if sdk == "mips" and not includes_mips_sdk(job_name, job_num):
        return False
    # cap-table is only valid for some builds:
    if isa == "cap-table":
        if not includes_isa_column(job_name, job_num):
            return False
        if tstruct != "0_256" or bitfile != sdk or sdk != target_arch or bitfile == "beri":
            return False
        return True

    if bitfile == "beri" and sdk == "mips" and target_arch == "mips":
        return True
    if bitfile == "cheri128" and sdk != "cheri256" and target_arch != "cheri256" and not (
            sdk == "mips" and target_arch == "cheri128"):
        return True
    if bitfile == "cheri256":
        if target_arch == "mips" and not includes_mips_on_256_bitfile(job_name, job_num):
            return False
        if sdk != "cheri128" and target_arch != "cheri128" and not (
                sdk == "mips" and target_arch == "cheri256"):
            return True
    return False


Config = namedtuple("Config", ("bitfile_cpu", "isa", "sdk_cpu", "target_arch_cpu", "tstruct", "job", "job_num"))
confs = [Config(bitfile, isa, sdk, target_arch, tstruct, job_name, job_num) for bitfile in bitfile_cpus
         for isa in ("vanilla", "cap-table")
         for sdk in sdk_cpus
         for target_arch in tgt_arch_cpus
         for tstruct in tstructs
         for job_name, e1 in jobs
         for job_num in e1
         if is_valid_job_config(bitfile, isa, sdk, target_arch, tstruct, job_name, job_num)
         ]  # type: typing.List[Conf]

for conf in confs:
    print(conf)
dfs = []
for bitfile_cpu, isa, sdk_cpu, tgt_arch_cpu, tstruct, job, jobnum in confs:
    if not includes_isa_column(job, jobnum):
        isa = None
    url = jenkins
    url += "/job/" + job
    if jobnum == LAST_SUCCESSFUL_JOB_NUM:
        # find out the real numerical id:
        api_cmd = base_cmd.copy()
        api_cmd.remove("-O")
        api_cmd.append(url + "/lastSuccessfulBuild/api/json")
        print(" ".join(api_cmd))
        output = sub.check_output(api_cmd)
        data = json.loads(output)
        # import pprint
        # pprint.pprint(data)
        jobnum = int(data["id"])
    filename = artifact_str(job, bitfile_cpu, isa, sdk_cpu, tgt_arch_cpu, tstruct, jobnum)
    url += "/{}/{}/artifact/".format(jobnum, jobargs_str(bitfile_cpu, isa, sdk_cpu, tgt_arch_cpu, tstruct))
    url += filename
    run_cmd = base_cmd + [url]
    print(" ".join(run_cmd))
    sub.check_call(run_cmd)
    data = pd.read_csv(filename)
    data.insert(0, 'table-struct', tstruct)
    if isa == "cap-table":
        tgt_arch_cpu += "-cap-table"
    data.insert(0, 'target-arch-cpu', tgt_arch_cpu)
    data.insert(0, 'sdk-cpu', sdk_cpu)
    data.insert(0, 'bitfile-cpu', bitfile_cpu)
    del data['archname']
    dfs.append(data)
df = pd.concat(dfs)
df.sort_values(["bitfile-cpu", "sdk-cpu", "target-arch-cpu", "table-struct", "progname"], inplace=True)
df.to_csv("hardware-results.csv", index=False)
