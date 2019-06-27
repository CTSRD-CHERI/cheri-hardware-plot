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
import subprocess
from enum import Enum
import argparse
import json
import subprocess as sub
import os
import pandas as pd
import shutil
import sys
from pathlib import Path
from collections import namedtuple


class Jobs(Enum):
    Olden = "bluehive-benchmark-olden"
    Mibench = "bluehive-benchmark-mibench"
    Duktape = "bluehive-benchmark-octane-duktape"
    Spec = "bluehive-benchmark-spec"
    InitDB = "bluehive-benchmark-postgres-initdb"


def jobargs_str(job, bitfile, isa: str, sdk, tgt_arch, tstruct):
    if job == Jobs.InitDB.value:
        # CPU=cheri128,ISA=cap-table-pcrel,LINKAGE=dynamic,label=bluehive/
        real_isa, linkage = isa.rsplit("-", maxsplit=1)
        return "CPU={},ISA={},LINKAGE={},label=bluehive/".format(tgt_arch, real_isa, linkage)
    isa_arg = ",ISA=" + isa if isa is not None else ""
    return "BITFILE_CPU={}{},SDK_CPU={},TARGET_ARCH_CPU={},TSTRUCT={},label=bluehive".format(bitfile, isa_arg, sdk,
                                                                                             tgt_arch, tstruct)


def artifact_str(job, bitfile, isa, sdk_cpu, tgt_arch_cpu, tstruct, jobnum):
    if job == Jobs.InitDB.value:
        # Same file name for all configurations
        return "postgres.statcounters.csv"
    isa_arg = "_ISA=" + isa if isa is not None else ""
    return "statcounters-jenkins-{}-BITFILE_CPU={}{}_SDK_CPU={}_TARGET_ARCH_CPU={}_TSTRUCT={}_label=bluehive-{}.csv".format(
        job, bitfile, isa_arg, sdk_cpu, tgt_arch_cpu, tstruct, jobnum)


def conf_str(a, b, c, d):
    return "{}-{}-{}-{}".format(a, b, c, d)


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--latest-only", action="store_true",
                    help="Only fetch the csv for the last successful run")
parser.add_argument("--subset", help="Subset of job csvs to fetch", default="19aug2017",
                    choices=["19aug2017", "cheriabi", "cap-table", "latest", "custom"])
jobs_lowercase = [job.name.lower() for job in Jobs]
parser.add_argument("--jobs", nargs=argparse.ONE_OR_MORE, help="Subset of job csvs to fetch",
                    choices=jobs_lowercase, default=jobs_lowercase)
parser.add_argument("--job-numbers", nargs=argparse.ONE_OR_MORE, help="List of job numbers to fetch")
parser.add_argument("--include-cheri256", action="store_true", help="Also fetch CHERI256 results")
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

curl = shutil.which("curl")
base_cmd = [curl, "--fail", "-sS", "-O", "-k", "-u", "{}:{}".format(user, password)]


def get_last_successful_run(job: Jobs) -> int:
    # find out the real numerical id:
    api_cmd = base_cmd.copy()
    api_cmd.remove("-O")
    api_cmd.append(jenkins + "/job/" + job.value + "/lastSuccessfulBuild/api/json")
    print(" ".join(api_cmd))
    output = sub.check_output(api_cmd)
    data = json.loads(output)
    # import pprint
    # pprint.pprint(data)
    jobnum = int(data["id"])
    print("Last successful", job.name, "run was", jobnum)
    return jobnum


# duktape_jobs = [113,114,115,116,117,118,119,120,125,126,127,128,129,130]
# olden_jobs   = [271,272,273] # don't use 274
# mibench_jobs = [234,235] # don't use 236 (matrix reloaded only certain sub-jobs)
# duktape_jobs = [145,153,161,162,167,168,169,173,174,176,177,185,192,193]
# olden_jobs   = [297,298]
# mibench_jobs = [244,245]
duktape_jobs = []
olden_jobs = []
mibench_jobs = []
spec_jobs = []
initdb_jobs = []

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
elif args.subset == "custom":
    duktape_jobs = [int(x) for x in args.job_numbers]
    mibench_jobs = [int(x) for x in args.job_numbers]
    olden_jobs = [int(x) for x in args.job_numbers]
    spec_jobs = [int(x) for x in args.job_numbers]
    initdb_jobs = [int(x) for x in args.job_numbers]

# Allow fetching only the latest job
if args.subset == "latest" or args.latest_only:
    if Jobs.Duktape.name.lower() in args.jobs:
        duktape_jobs = [get_last_successful_run(Jobs.Duktape)]
    if Jobs.Olden.name.lower() in args.jobs:
        olden_jobs = [get_last_successful_run(Jobs.Olden)]
    if Jobs.Mibench.name.lower() in args.jobs:
        mibench_jobs = [get_last_successful_run(Jobs.Mibench)]
    if Jobs.Spec.name.lower() in args.jobs:
        spec_jobs = [get_last_successful_run(Jobs.Spec)]
    if Jobs.InitDB.name.lower() in args.jobs:
        initdb_jobs = [get_last_successful_run(Jobs.InitDB)]

# XXXAR: no idea what the actual job number is where this was changed, let's just use something
def includes_tstruct_0(job, job_num):
    if job == Jobs.Spec.value:
        return False
    # TODO: find the right numbers
    return job_num < 400


def includes_mips_sdk(job, job_num):
    if job == Jobs.Spec.value:
        return False
    # TODO: find the right numbers
    return job_num < 400


def includes_mips_on_256_bitfile(job, job_num):
    if job == Jobs.Spec.value:
        return False
    # most runs use the cheri128 bitfile
    # TODO: do we need this?
    return job_num < 400


def includes_beri_bitfile(job, job_num):
    # most runs use the cheri128 bitfile for running MIPS (for memcpy performance)
    return job != Jobs.Spec.value and job_num < 400

def includes_isa_column(job, job_num):
    if job == Jobs.Olden.value:
        return job_num > 550
    elif job == Jobs.Mibench.value:
        return job_num > 493
    elif job == Jobs.Duktape.value:
        return False
    elif job == Jobs.Spec.value or job == Jobs.InitDB.value:
        return True
    assert False, "Bad job " + job


def includes_nobounds_isa(job, job_num):
    if job == Jobs.Olden.value:
        return job_num >= 569
    if job == Jobs.Mibench.value:
        return job_num > 511
    elif job == Jobs.Duktape.value:
        return False
    assert False, "Bad job " + job


jobs = []
if Jobs.Duktape.name.lower() in args.jobs:
    jobs += [(Jobs.Duktape.value, duktape_jobs)]
if Jobs.Olden.name.lower() in args.jobs:
    jobs += [(Jobs.Olden.value, olden_jobs)]
if Jobs.Mibench.name.lower() in args.jobs:
    jobs += [(Jobs.Mibench.value, mibench_jobs)]
if Jobs.Spec.name.lower() in args.jobs:
    jobs += [(Jobs.Spec.value, spec_jobs)]

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


def fixup_initdb_data(df: pd.DataFrame):
    # Initdb spawns 5 postgres processes, rename those to have a different progname so we can measure all
    for i, row in df.iterrows():
        progname = str(row['progname'])
        count = i % 6
        if progname == 'postgres':
            assert count != 5, "every sixth row should have name initdb"
            new_progname = progname + "-child-" + str(count + 1)
            df.set_value(i, 'progname', new_progname)
        else:
            # the final row of each run should be initdb
            assert count == 5, count
            assert progname == "initdb", progname
    return df


def is_valid_job_config(bitfile, isa, sdk, target_arch, tstruct, job_name, job_num):
    # Only pcrel for spec
    if job_name == Jobs.Spec.value:
        if isa != "cap-table-pcrel":
            return False
        print(locals())
    elif job_name == Jobs.Olden.value and job_num >= 900:
        # No more vanilla, nobounds
        # only includes legacy legacy-nobounds cap-table-pcrel
        if isa not in ("legacy", "legacy-nobounds", "cap-table-pcrel"):
            return False
    elif job_name == Jobs.Mibench.value and job_num >= 1200:
        # No more vanilla, nobounds
        # only includes legacy legacy-nobounds cap-table-pcrel
        if isa not in ("legacy", "mips-asan", "cap-table-pcrel"):
            return False
    # cap-table is only valid for some builds:
    elif isa in ("cap-table", "nobounds"):
        if not includes_isa_column(job_name, job_num):
            return False
        if isa == "nobounds" and not includes_nobounds_isa(job_name, job_num):
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


# This is insanely inefficient if we ever end up with more than a few job configs:
Config = namedtuple("Config", ("bitfile_cpu", "isa", "sdk_cpu", "target_arch_cpu", "tstruct", "job", "job_num"))
confs = []  # type: typing.List[Conf]
for bitfile in bitfile_cpus:
    if bitfile == "cheri256" and not args.include_cheri256:
        continue
    for job_name, job_numbers in jobs:
        for job_num in job_numbers:
            # we run mips on cheri128 starting after some job
            if bitfile == "beri" and not includes_beri_bitfile(job_name, job_num):
                continue
            for isa in ("legacy", "legacy-nobounds", "vanilla", "cap-table", "cap-table-pcrel", "nobounds"):
                for sdk in sdk_cpus:
                    if sdk == "cheri256" and not args.include_cheri256:
                        continue
                    if bitfile == "beri" and sdk.startswith("cheri"):
                        continue
                    if bitfile.startswith("cheri") and sdk.startswith("cheri") and bitfile != sdk:
                        continue  # can't run cheri128 on cheri256 bitfile
                    # no more MIPS sdk builds after some build:
                    if sdk == "mips" and not includes_mips_sdk(job_name, job_num):
                        continue
                    for target_arch in tgt_arch_cpus:
                        if target_arch == "cheri256" and not args.include_cheri256:
                            continue
                        if sdk.startswith("cheri") and target_arch.startswith("cheri") and sdk != target_arch:
                            continue  # can't run cheri128 on cheri256 bitfile
                        if bitfile == "cheri256" and target_arch == "mips" and not includes_mips_on_256_bitfile(job_name, job_num):
                            continue
                        for tstruct in tstructs:
                            if bitfile == "beri" and tstruct == "0_256":
                                continue
                            # no more tstruct 0 for CHERI builds after some job#
                            if tstruct == "0" and not includes_tstruct_0(job_name, job_num):
                                continue
                            if is_valid_job_config(bitfile, isa, sdk, target_arch, tstruct, job_name, job_num):
                                confs.append(Config(bitfile, isa, sdk, target_arch, tstruct, job_name, job_num))


# Postgres uses a different naming scheme:
if Jobs.InitDB.name.lower() in args.jobs:
    archs = ["mips", "mips-asan", "cheri128"]
    # only cap-table-pcrel running:
    isa_base = "cap-table-pcrel"
    tstruct = "0_256"
    job_name = Jobs.InitDB.value
    if args.include_cheri256:
        archs.append("cheri256")
    for job_num in initdb_jobs:
        for linkage in ("static", "dynamic"):
            isa = isa_base + "-" + linkage
            # CPU=cheri128,ISA=cap-table-pcrel,LINKAGE=dynamic,label=bluehive/
            for target_arch in archs:
                if target_arch.startswith("mips"):
                    sdk = "cheri128"
                    bitfile = "cheri128"
                else:
                    sdk = target_arch
                    bitfile = target_arch
                confs.append(Config(bitfile, isa, sdk, target_arch, tstruct, job_name, job_num))

assert confs
for conf in confs:
    print(conf)
dfs = []
for bitfile_cpu, isa, sdk_cpu, tgt_arch_cpu, tstruct, job, jobnum in confs:
    if not includes_isa_column(job, jobnum):
        isa = None
    url = jenkins
    url += "/job/" + job
    filename = artifact_str(job, bitfile_cpu, isa, sdk_cpu, tgt_arch_cpu, tstruct, jobnum)
    url += "/{}/{}/artifact/".format(jobnum, jobargs_str(job, bitfile_cpu, isa, sdk_cpu, tgt_arch_cpu, tstruct))
    url += filename
    run_cmd = base_cmd + [url]
    print(" ".join(run_cmd))
    try:
        sub.check_call(run_cmd)
    except subprocess.CalledProcessError:
        print("WARNING: Could not find a csv file for configuration",
              "bitfile_cpu={}, isa={}, sdk_cpu={}, tgt_arch_cpu={}, tstruct={}, job={}, jobnum={}".format(
                  bitfile_cpu, isa, sdk_cpu, tgt_arch_cpu, tstruct, job, jobnum), file=sys.stderr)
        if input("Continue? y/n").lower().startswith("y"):
            continue
        else:
            sys.exit(1)
    out_path = Path(filename)
    if out_path.stat().st_size == 0:
        print("WARNING: Could not find a csv file for configuration",
              "bitfile_cpu={}, isa={}, sdk_cpu={}, tgt_arch_cpu={}, tstruct={}, job={}, jobnum={}".format(
                  bitfile_cpu, isa, sdk_cpu, tgt_arch_cpu, tstruct, job, jobnum), file=sys.stderr)
        Path(filename).unlink()
        continue
    data = pd.read_csv(filename)
    if job == Jobs.InitDB.value:
        data = fixup_initdb_data(data)
        # keep all the temporary files:
        out_path.rename(out_path.with_name(out_path.stem + "-" + tgt_arch_cpu + "-" + isa + "-" + str(job_num) + ".csv"))
    data.insert(0, 'table-struct', tstruct)
    if isa is not None and isa != "vanilla":
        tgt_arch_cpu += "-" + isa
    data.insert(0, 'target-arch-cpu', tgt_arch_cpu)
    data.insert(0, 'sdk-cpu', sdk_cpu)
    data.insert(0, 'bitfile-cpu', bitfile_cpu)
    del data['archname']
    dfs.append(data)
df = pd.concat(dfs)
df.sort_values(["bitfile-cpu", "sdk-cpu", "target-arch-cpu", "table-struct", "progname"], inplace=True)
df.to_csv("hardware-results.csv", index=False)
