import os

INPUT_DIR = "D:/Documents/SAIL/musicoder-implementation/data/mtg_first_20"
OUTPUT_DIR = "D:/Documents/SAIL/musicoder-implementation/data"
NUM_TFRECORDS = 4
TXT_FILE = "print_tfrecord_cmd.txt"

files = []
for filename in os.listdir(INPUT_DIR):
    if filename.endswith(".npy"):
        files.append(filename)

cmd = "python create_pretraining_data.py --num_features 96 --max_seq_length 1000"

# input files
cmd+=" --input_file "
cmd+=INPUT_DIR+"/*"
# for f in files:
#     cmd+=INPUT_DIR+"/"+f+","
# cmd = cmd[:-1]

# output files

cmd+=" --output_file "
for i in range(NUM_TFRECORDS):
    cmd+=OUTPUT_DIR+"/"+"first20_"+str(i)+".tfrecord,"
cmd = cmd[:-1]

print(cmd)
with open(TXT_FILE, "w") as f:
    f.write(cmd)