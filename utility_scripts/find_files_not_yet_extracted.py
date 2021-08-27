# Create a TXT file of all files that are in input_dir (recursive search) but not in output_dir.

import argparse
import os
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--input_dir", help="input directory", required=True)
parser.add_argument("--output_dir", help="output directory", required=True)
parser.add_argument("--output_file", help="output file", required=True)
args = parser.parse_args()

total = []
for (dirpath, dirnames, filenames) in os.walk(args.input_dir):
    total += [ (file[:-4], os.path.join(dirpath, file)) for file in filenames if file[-4:]==(".mp3")]

done = []
for (dirpath, dirnames, filenames) in os.walk(args.output_dir):
    done += [ file[:-4] for file in filenames if file[-4:]==(".npy")]

left = []
for x in tqdm(total):
    if len(x) != 2:
        print("Alert! {} is not a tuple".format(x))
    name = x[0]
    path = x[1]
    if name not in done:
        left.append(path)

with open(args.output_file, "w") as f:
    for p in left:
        f.write(p+"\n")
