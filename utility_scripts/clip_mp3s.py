import argparse
from pydub import AudioSegment
from pydub.utils import mediainfo
import os
from tqdm import tqdm

# Grab params
parser = argparse.ArgumentParser()
parser.add_argument("--input_dir", help="input directory in which to clip all mp3s")
parser.add_argument("--output_dir", help="output directory to output mp3s to")
parser.add_argument("--clip_length", help="length of the resulting clips (in seconds)", type=int)
parser.add_argument("--output_bitrate",
                    help="output bitrate, to keep original use 'original' otherwise, put the kbps desired (e.g. '128')",
                    default="original")
args = parser.parse_args()

# Iterate over all files in input_dir...
files = []
for filename in os.listdir(args.input_dir):
    if filename.endswith(".mp3"):
        files.append( (os.path.join(args.input_dir, filename), filename) ) # (path, file)
for (path, file) in tqdm(files):
    raw = AudioSegment.from_mp3(path)
    if args.output_bitrate == "original":
        bitrate = mediainfo(path)['bit_rate']
    else:
        bitrate = args.output_bitrate+"k"

    # Find middle clip_length seconds
    start_time = 0
    end_time = len(raw)
    if end_time > args.clip_length*1000: # in milliseconds
        middle = end_time // 2
        start_time = middle - (args.clip_length/2) * 1000
        end_time = middle + (args.clip_length / 2) * 1000
    else:
        print("Warning: Mp3 clip '{}' of length {} seconds is shorter than desired clip length ({} seconds).".format(
            path, end_time/1000, args.clip_length
        ))

    # Clip and export
    # create output dir if not extant
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    raw[start_time:end_time].export( os.path.join(args.output_dir, file), format="mp3", bitrate=bitrate )