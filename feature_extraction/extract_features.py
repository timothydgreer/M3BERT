# Python 3.x
# Extracts features from all mp3s found in all directories and subdirectories of INPUT_FOLDER, outputting features
# to OUTPUT_FOLDER

import librosa
import numpy as np
import os
from speechpy import processing
import ntpath
import argparse
from tqdm import tqdm
import random

parser = argparse.ArgumentParser()
parser.add_argument("--input_dir", help="input directory in which to clip all mp3s", required=True)
parser.add_argument("--output_dir", help="output directory to output mp3s to", required=True)
parser.add_argument("--output_dtype", help="output dtype. supported: float16, float32, float64", default="float16")
parser.add_argument("--overwrite", help="whether to overwrite files that already exist. default=False", default=False, action="store_true")
args = parser.parse_args()

INPUT_FOLDER = args.input_dir
OUTPUT_FOLDER = args.output_dir
if args.output_dtype == "float16":
    OUTPUT_DTYPE = np.float16
elif args.output_dtype == "float32":
    OUTPUT_DTYPE = np.float32
elif args.output_dtype == "float64":
    OUTPUT_DTYPE = np.float64
else:
    raise ValueError("Provided output_dtype {} not supported. Supported: float16, float32, float64".format(args.output_dtype))

# SETTINGS
sr = 44100
window = 'hamming'
win_length=2048
hop_length=1024

def log_scale(x):
    epsilon = 1e-6
    return (np.log(10*x+epsilon))

# Find all mp3 files in dir and subdirs
mp3s = []
for (dirpath, dirnames, filenames) in os.walk(INPUT_FOLDER):
    mp3s += [os.path.join(dirpath, file) for file in filenames if file[-4:]==(".mp3")]

# shuffle mp3 order (in case multiple processes running, don't want them having same exact order)
random.shuffle(mp3s)

(maxes, mins) = (None, None)
num_feats = None

# Process every file found
i = 0
for mp3 in tqdm(mp3s):
    try:
        # Skip if overwrite=False and output file already exists
        output_file = os.path.join(OUTPUT_FOLDER, ntpath.basename(mp3)).replace(".mp3", ".npy")
        if os.path.exists(output_file) and not args.overwrite:
            print("Skipping {} as it already exists in output destination".format(ntpath.basename(mp3)))
            continue

        x, sr = librosa.load(mp3, sr=sr)

        # Extract features
        mel_raw = librosa.feature.melspectrogram(y=x, sr=sr, hop_length=hop_length, win_length=win_length, window=window)
        cqt_raw = librosa.feature.chroma_cqt(y=x, sr=sr, hop_length=hop_length, n_chroma=144, bins_per_octave=None)
        mfcc_raw = librosa.feature.mfcc(y=x, sr=sr, hop_length=hop_length, win_length=win_length, window=window)
        delta_mfcc_raw = librosa.feature.delta(mfcc_raw)
        chroma_raw = librosa.feature.chroma_stft(y=x, sr=sr, hop_length=hop_length, win_length=win_length, window=window)

        # Perform log-scaling and cepstral mean variance normalization
        mel = processing.cmvn(log_scale(mel_raw))
        cqt = processing.cmvn(log_scale(cqt_raw))
        mfcc = processing.cmvn(mfcc_raw)
        delta_mfcc = processing.cmvn(delta_mfcc_raw)
        chroma = processing.cmvn(chroma_raw)

        # Stack
        output = np.vstack( (chroma, mfcc, delta_mfcc, mel, cqt) ).T.astype(OUTPUT_DTYPE) # order described in Musicoder paper Table 3

        # Keep track of max/min (for column-wise normalization later)
        if maxes is None:
            num_feats = output.shape[1]
            maxes = np.ones((num_feats,)) * -np.inf
            mins = np.ones((num_feats,)) * np.inf
        maxes = np.max(np.vstack((output, maxes)), axis=0)
        mins = np.min(np.vstack((output, mins)), axis=0)

        # Create output dir if not extant
        if not os.path.exists(OUTPUT_FOLDER):
            os.mkdir(OUTPUT_FOLDER)

        # Output mp3

        np.save(output_file, output)

        # Update maxes and mins every 1000 mp3s
        i+=1
        if i%1000==0:
            # compare to saved maxes/mins if exist
            if os.path.exists(os.path.join(OUTPUT_FOLDER, "maxes.npy")):
                maxes_prev = np.load(os.path.join(OUTPUT_FOLDER, "maxes.npy"))
                mins_prev = np.load(os.path.join(OUTPUT_FOLDER, "mins.npy"))
            maxes = np.max(np.vstack((maxes_prev, maxes)), axis=0)
            mins = np.min(np.vstack((mins_prev, mins)), axis=0)
            # Output column-wise maxes/mins
            np.save(os.path.join(OUTPUT_FOLDER, "maxes.npy"), maxes)
            np.save(os.path.join(OUTPUT_FOLDER, "mins.npy"), mins)
    except Exception as e:
        print("{} on mp3 {}: {}".format(type(e), ntpath.basename(mp3), e))
        continue

# Output column-wise maxes/mins
np.save(os.path.join(OUTPUT_FOLDER, "maxes.npy"), maxes)
np.save(os.path.join(OUTPUT_FOLDER, "mins.npy"), mins)

