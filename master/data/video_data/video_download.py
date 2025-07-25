from __future__ import absolute_import, division, print_function
import os
import datetime
import pandas as pd
import time
import sys

sys.path.insert(0, 'Looking-to-Listen-at-the-Cocktail-Party-master/data/utils')
import utils

def video_download(loc, d_csv, start_idx, end_idx):
    utils.mkdir(loc)
    for i in range(start_idx, end_idx):
        link = "https://www.youtube.com/watch?v=" + d_csv.loc[i][0]
        f_name = os.path.join(loc, str(i))
        start_time = d_csv.loc[i][1]
        end_time = start_time + 3.0
        start_time_str = str(datetime.timedelta(seconds=start_time))
        end_time_str = str(datetime.timedelta(seconds=end_time))

        yt_command = f'yt-dlp -f "mp4" -o "{f_name}_temp.mp4" {link}'
        ffmpeg_command = f'ffmpeg -i "{f_name}_temp.mp4" -c:v h264 -c:a copy -ss {start_time_str} -to {end_time_str} "{f_name}.mp4"'

        os.system(yt_command)
        os.system(ffmpeg_command)
        os.remove(f"{f_name}_temp.mp4")

def generate_frames(loc, start_idx, end_idx):
    frames_dir = os.path.abspath("frames")  # Use absolute path to avoid issues
    utils.mkdir(frames_dir)
    for i in range(start_idx, end_idx):
        f_name = os.path.join(loc, str(i))
        frame_command = f'ffmpeg -i "{f_name}.mp4" -vf fps=25 "{frames_dir}/{i}-%02d.jpg"'
        os.system(frame_command)

def download_video_frames(loc, d_csv, start_idx, end_idx, rm_video=False):
    frames_dir = os.path.abspath("frames")
    utils.mkdir(frames_dir)
    utils.mkdir(loc)
    
    for i in range(start_idx, end_idx):
        link = "https://www.youtube.com/watch?v=" + d_csv.loc[i][0]
        f_name = os.path.join(loc, str(i))
        start_time = d_csv.loc[i][1]
        start_time_str = time.strftime("%H:%M:%S.0", time.gmtime(start_time))

        yt_command = f'yt-dlp -f "mp4" -o "{f_name}_temp.mp4" {link}'
        ffmpeg_video_command = f'ffmpeg -i "{f_name}_temp.mp4" -c:v h264 -c:a copy -ss {start_time_str} -t 3 "{f_name}.mp4"'
        ffmpeg_frames_command = f'ffmpeg -i "{f_name}.mp4" -vf fps=25 "{frames_dir}/{i}-%02d.jpg"'

        os.system(yt_command)

        if os.path.exists(f"{f_name}_temp.mp4"):
            os.system(ffmpeg_video_command)
            os.system(ffmpeg_frames_command)

            if rm_video and os.path.exists(f"{f_name}.mp4"):
                os.remove(f"{f_name}.mp4")
            os.remove(f"{f_name}_temp.mp4")
        else:
            print(f"Download failed for video {i} - skipping")

        print(f"\rProcessed video {i}", end="")
    print("\rFinished!!", end="")

# Load CSV and download videos
utils.mkdir('video_train')
cat_train = pd.read_csv('Looking-to-Listen-at-the-Cocktail-Party-master/data/csv/avspeech_test.csv', header=None)
download_video_frames(loc='video_train', d_csv=cat_train, start_idx=0, end_idx=19, rm_video=False)
