import os
import librosa
import scipy.io.wavfile as wavfile
import numpy as np


def mkdir(location):
    folder = os.path.exists(location)
    if not folder:
        os.mkdir(location)
        print("mkdir "+location+" ——success")
    else:
        print("location folder exists!!")


def download(loc, name, link, sr=16000, file_type='audio'):
    if file_type == 'audio':
        # Change to target directory
        original_dir = os.getcwd()
        os.chdir(loc)

        # Construct Windows-compatible command
        yt_dlp_command = f'yt-dlp -x --audio-format wav -o o{name}.wav {link}'
        ffmpeg_command = f'ffmpeg -i o{name}.wav -ar {sr} -ac 1 {name}.wav'
        delete_command = f'del o{name}.wav'  # Windows delete command

        # Run yt-dlp, ffmpeg, and delete in sequence
        os.system(yt_dlp_command)
        os.system(ffmpeg_command)
        os.system(delete_command)

        # Return to the original directory
        os.chdir(original_dir)


def cut(loc, name, start_time, end_time):
    # Ensure the correct paths are used and check if the source file exists
    source_file = os.path.join(loc, f"{name}.wav")
    if not os.path.exists(source_file):
        print(f"Error: The source file '{source_file}' does not exist.")
        return
    
    length = end_time - start_time
    trimmed_file = os.path.join(loc, f"trim_{name}.wav")

    # Construct the sox command for trimming audio
    command = f'sox "{source_file}" "{trimmed_file}" trim {start_time} {length}'
    
    # Execute the command
    exit_code = os.system(command)

    # Check if the trimming was successful
    if exit_code != 0:
        print(f"Error: Failed to trim audio file '{source_file}'. Command: {command}")
    else:
        print(f"Successfully trimmed audio '{source_file}' to '{trimmed_file}'.")

    # Optionally, remove the original file if you want
    os.remove(source_file)


def conc(loc, name, trim_clean=False):
    # Concatenate the data in the loc (trim*.wav)
    command = f'sox "{loc}/trim_*.wav" -o "{loc}/{name}.wav"'
    if trim_clean:
        command += ' && del "trim_*.wav"'
    
    os.system(command)


def mix(loc, name, file1, file2, start, end, trim_clean=False):
    cut(loc, file1, start, end)
    cut(loc, file2, start, end)
    trim1 = os.path.join(loc, f"trim_{file1}.wav")
    trim2 = os.path.join(loc, f"trim_{file2}.wav")

    with open(trim1, 'rb') as f:
        wav1, wav1_sr = librosa.load(trim1, sr=None)  # time series data,sample rate
    with open(trim2, 'rb') as f:
        wav2, wav2_sr = librosa.load(trim2, sr=None)

    # Compress the audio to the same volume level
    wav1 = wav1 / np.max(wav1)
    wav2 = wav2 / np.max(wav2)
    assert wav1_sr == wav2_sr
    mix_wav = wav1 * 0.5 + wav2 * 0.5

    path = os.path.join(loc, f"{name}.wav")
    wavfile.write(path, wav1_sr, mix_wav)
    
    if trim_clean:
        os.remove(trim1)
        os.remove(trim2)
