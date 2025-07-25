import sys
import os
import pandas as pd

# Adjust path for importing utils
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, 'Looking-to-Listen-at-the-Cocktail-Party-master/data/utils')
import utils

def make_audio(location, name, d_csv, start_idx, end_idx):
    # Ensure the location folder exists
    audio_dir = os.path.abspath(location)
    os.makedirs(audio_dir, exist_ok=True)
    
    for i in range(start_idx, end_idx):
        f_name = f"{name}{i}"
        link = f"https://www.youtube.com/watch?v={d_csv.loc[i][0]}"
        start_time = d_csv.loc[i][1]
        end_time = start_time + 3.0

        try:
            # Download the audio
            utils.download(location, f_name, link)
            
            # Verify the audio file exists before cutting
            if not os.path.exists(os.path.join(audio_dir, f"{f_name}.wav")):
                print(f"Error: Audio file {f_name}.wav not found after downloading.")
                continue

            # Cut the audio
            utils.cut(audio_dir, f_name, start_time, end_time)
            print(f"\rProcessing audio... {i}", end="")
        except Exception as e:
            print(f"\nError processing audio for index {i}: {e}")
            
    print("\rFinished processing all audios.", end="")

# Ensure output directory exists
output_dir = 'audio_train'
os.makedirs(output_dir, exist_ok=True)

# Load CSV data for processing
cat_train = pd.read_csv('Looking-to-Listen-at-the-Cocktail-Party-master/data/csv/avspeech_test.csv', header=None)

# Process audio data for a subset of records (adjust start_idx and end_idx as needed)
make_audio(output_dir, 'audio_train', cat_train, start_idx=0, end_idx=20)
