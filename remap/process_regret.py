import pandas
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import tempfile
from pathlib import Path
import subprocess
import regex

def parse_args():
    parser = argparse.ArgumentParser(description='Provide Filenames')
    parser.add_argument('--ref', type=str, default=None,
                        help='Specifies the path to the reference data')
    parser.add_argument('--learned', type=str, action="append", default=None,
                        help='Specifies the path to the learned data')
    parser.add_argument('--var', type=int, action="append", default=None,
                        help='List of dependent variables')
    parser.add_argument('--save', type=str, default=None,
                        help='Path to save the resulting data to')
    parser.add_argument('--lower', type=int, default=1e6,
                        help='Lower bound of window')
    parser.add_argument('--upper', type=int, default=2e6,
                        help='Upper bound of window')
    args = parser.parse_args()
    return args

def check_seed(path, upper_bound):
    """
    Given a directory path, check that the proper information is available in the directory,
    and validate that the run for that seed finished successfully.

    If we have progress.csv, and it contains as its last line something like 45217,2000000,284.0,
    then we are good to go with using this seed

    Returns True if this seed should be used, and False otherwise
    """
    progress = Path(path) / "progress.csv"
    if progress.is_file():
        ## Check if the last line is the appropriate format
        p = subprocess.run(["/usr/bin/tail", "-n", "1", progress], capture_output=True, text=True)
        return f",{upper_bound}," in p.stdout
    else:
        return False

def find_seeds(base, pattern):
    """
    Given a base diretory, find the seed sub directories matching a specific list of patterns
    """
    re = regex.compile(pattern[0])
    for element in Path(base).iterdir():
        if re.fullmatch(element.name):
            if len(pattern) > 1:
                yield from find_seeds(element, pattern[1:])
            else:
                yield element

def read_progress(seed, lower_bound, upper_bound):
    csv = Path(seed) / "progress.csv"
    data = None
    with csv.open() as f:
        data = f.read().splitlines()
    ## Ignore the header
    ## Find the index of the lower bound
    ratio = lower_bound / upper_bound
    ## Starting index
    idx = int(len(data)*ratio - 1)
    while f",{lower_bound}," not in data[idx]:
        idx += 1
    ## Now data[idx] contains the lower_bound data. Therefore, we just need to start on idx+1
    total_sum = sum(float(element.split(",")[-1]) for element in data[idx+1:])
    return total_sum / (upper_bound - lower_bound)
        
def get_empirical_avg(target, pattern, lower, upper):
    total_used_seeds = 0
    total_avg_reward_per_step = 0
    list_avg_per_step = list()
    for seed in find_seeds(target, pattern):
        if check_seed(seed, upper):
            total_used_seeds += 1
            avg_reward_per_step = read_progress(seed, lower, upper)
            total_avg_reward_per_step += avg_reward_per_step
            list_avg_per_step.append(avg_reward_per_step)

    ## Empirical Average:
    mu = total_avg_reward_per_step / total_used_seeds
    ## Compute sample variance
    sample_var = sum((el-mu)*(el-mu) for el in list_avg_per_step)/(total_used_seeds-1)
    return mu, sample_var

def main():
    args = parse_args()
    ## Under the args.ref path, there will be N seeds. Within each seed, we need to check if
    ## it completed successfully.
    print(f"Processing {args.ref}...")
    ## Sample mean and sample variance
    ref_avg, ref_var = get_empirical_avg(args.ref, ["\d+"], args.lower, args.upper)
    
    regret_avgs = list()
    regret_vars = list()
    for learned in args.learned:
        print(f"Processing {learned}...")
        l_avg, l_var = get_empirical_avg(learned, ["TRIAL-\d+","\d+"], args.lower, args.upper)
        regret_avgs.append(ref_avg - l_avg)
        ## Sample variance is the sum of the variances
        regret_vars.append(ref_var + l_var)

    print(f"Saving results to {args.save}...")
    with open(args.save, "w") as f:
        f.write(f"'Variable'#'Avg Regret'#'Variance'\n")
        for idx in range(len(regret_avgs)):
            f.write(f"{args.var[idx]}#{regret_avgs[idx]}#{regret_vars[idx]}\n")

if __name__ == "__main__":
    main()
