# Ben Ma, Python 3.x
# This script normalizes all .npy files in a given directory and its subdirectories, column-wise.
# Note that all .npy files should have the same shape (N x seq_length x num_feats) or (seq_length x num_feats)
# The script outputs all normalized .npy files to OUTPUT_DIR
# This script is designed to still work even if the total data of the .npy files together exceeds system memory

import os
import numpy as np
from tqdm import tqdm
import gc

INPUT_DIR = "D:\Downloads\mtg_jamendo"
OUTPUT_DIR = "D:\Documents\SAIL\musicoder-implementation\data\mtg_first_20"
TRANSPOSE = True # whether to transpose numpy files when opening. has no effect if file is 3D
NUM_FEATS = 96

# Step 0 ---- get names of all .npy files in dir and subdirs
input_files = []
for (dirpath, dirnames, filenames) in os.walk(INPUT_DIR):
    input_files += [os.path.join(dirpath, file) for file in filenames if file.endswith(".npy")]

# Step 1 ---- Find min and max of each column across all files
maxes = np.ones((NUM_FEATS,)) * -np.inf
mins = np.ones((NUM_FEATS,)) * np.inf

print("Finding MAX/MIN: ")
for i in tqdm(range(len(input_files))):
    f = input_files[i]
    arr = np.load(f)
    if len(arr.shape) == 2:
        if TRANSPOSE:
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

# Step 2 ---- Normalize each file and output
print("Normalizing and outputting: ")
for i in tqdm(range(len(input_files))):
    f = input_files[i]
    arr = np.load(f)
    if len(arr.shape) == 2:
        if TRANSPOSE:
            arr = arr.T
        arr = np.expand_dims(arr, 0)  # convert to 3D by adding dim of size 1
    normed = ((arr - mins) / (maxes - mins))
    np.save(OUTPUT_DIR+"/"+str(i)+".npy", normed)
    if i % 500 == 0:
        gc.collect() # run garbage collector