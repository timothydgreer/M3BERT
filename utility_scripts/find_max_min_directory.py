# Ben Ma, Python 3.x
# This script finds maxes and mins of all .npy files in a given directory and its subdirectories, column-wise.
# Note that all .npy files should have the same shape (N x seq_length x num_feats) or (seq_length x num_feats)
# The script outputs maxes.npy, mins.npy files to OUTPUT_DIR
# This script is designed to still work even if the total data of the .npy files together exceeds system memory

import os
import numpy as np
from tqdm import tqdm
import gc
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--input_dir", help="input directory to read .npy files from")
parser.add_argument("--output_dir", help="output directory to output maxes.npy, mins.npy to")
parser.add_argument("--num_feats", help="expected number of features (columns)", type=int)
parser.add_argument("--transpose", help="whether to transpose matrices when opening. default=False", default=False, action="store_true")
args = parser.parse_args()

# Step 0 ---- get names of all .npy files in dir and subdirs
input_files = []
for (dirpath, dirnames, filenames) in os.walk(args.input_dir):
    input_files += [os.path.join(dirpath, file) for file in filenames if (file.endswith(".npy")
                    and file != "maxes.npy" and file != "mins.npy")]

# Find min and max of each column across all files
maxes = np.ones((args.num_feats,)) * -np.inf
mins = np.ones((args.num_feats,)) * np.inf

print("Finding MAX/MIN: ")
for i in tqdm(range(len(input_files))):
    f = input_files[i]
    try:
        arr = np.load(f)
        if len(arr.shape) == 2:
            if args.transpose:
                arr = arr.T
            arr = np.expand_dims(arr, 0) # convert to 3D by adding dim of size 1
        maxes = (
            np.vstack((
                arr.max(axis=(0,1)), # (NUM_FEATS,)
                maxes # (NUM_FEATS,)
            ))
        ).max(axis=0)
        mins = (
            np.vstack((
                arr.min(axis=(0, 1)),  # (NUM_FEATS,)
                mins  # (NUM_FEATS,)
            ))
        ).min(axis=0)
        if i % 500 == 0:
            gc.collect() # run garbage collector
    except Exception as e:
        print("{} on file {}: {}".format(type(e), f, e))

# Output

# create output dir if not extant
if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)
np.save(os.path.join(args.output_dir, "maxes.npy"), maxes)
np.save(os.path.join(args.output_dir, "mins.npy"), mins)