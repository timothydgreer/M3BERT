import os

for i in range(0, 100):
    folder = str(i) if i>9 else "0"+str(i)

    # if this folder's tar file is present, go for it (extract, clip, and delete)
    tar_file = "raw_30s_audio-"+folder+".tar"
    if os.path.exists(tar_file):
        extract_cmd = "tar -xvf raw_30s_audio-"+folder+".tar "+folder
        process_cmd = "/opt/anaconda/anaconda_2019_10/bin/python3 /home/greert/Desktop/musicoder/mtg-jamendo-dataset/audio/clip_mp3.py --input_dir "+folder+" --output_dir "+folder+"_clip " \
                    "--clip_length 30 --output_bitrate 128"
        delete_cmd = "rm raw_30s_audio-"+folder+".tar && rm -r "+folder
        os.system(extract_cmd+" && "+process_cmd+" && "+delete_cmd)