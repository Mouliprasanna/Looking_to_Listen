import os
import subprocess

def run_command(command):
    """Runs a shell command and checks if it was successful."""
    result = subprocess.run(command, shell=True)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed: {command}")

def setup_environment():
    """Set up the environment by installing dependencies."""
    # Install required Python packages
    if not os.path.exists("requirements_installed.txt"):
        print("Installing Python dependencies...")
        run_command("pip install -r requirements.txt")
        with open("requirements_installed.txt", "w") as f:
            f.write("Dependencies installed.")

    # Install ffmpeg and sox
    try:
        run_command("ffmpeg -version")
    except RuntimeError:
        print("Installing ffmpeg...")
        run_command("brew install ffmpeg")

    try:
        run_command("sox --version")
    except RuntimeError:
        print("Installing sox...")
        run_command("brew install sox")

    # Install yt-dlp
    print("Installing yt-dlp...")
    run_command("pip install yt-dlp")

def download_data():
    """Download the required datasets."""
    # Download model checkpoint files
    if not os.path.exists("base/model.ckpt"):
        print("Downloading model checkpoint files...")
        run_command("curl -L -o base/model.ckpt 'https://drive.usercontent.google.com/download?id=1EXPBSXwTaqrSC0OhUdXNmKSh9qJUQ55-&export=download' ")

    # Download AVSpeech dataset
    if not os.path.exists("Looking-to-Listen-at-the-Cocktail-Party-master/data/csv/avspeech-dataset.csv"):
        print("Downloading AVSpeech dataset...")
        # Example: Replace the actual URL with the appropriate script for dataset download
        run_command("curl -L -o Looking-to-Listen-at-the-Cocktail-Party-master/data/csv/avspeech-dataset.csv 'https://looking-to-listen.github.io/avspeech/download.html'")

def preprocess_data():
    """Preprocess the video and audio data."""
    print("Preprocessing video data...")
    #run_command("python3 Looking-to-Listen-at-the-Cocktail-Party-master/data/video_data/video_download.py")
    run_command("pip install mtcnn")
    #run_command("python3 Looking-to-Listen-at-the-Cocktail-Party-master/data/video_data/face_detect.py")
    #run_command("python3 Looking-to-Listen-at-the-Cocktail-Party-master/data/video_data/check_valid_face.py")

    print("Preprocessing audio data...")
    #run_command("python3 Looking-to-Listen-at-the-Cocktail-Party-master/data/audio_data/audio_downloads.py")
    run_command("python3 Looking-to-Listen-at-the-Cocktail-Party-master/data/audio_data/audio_norm.py")
    run_command("python3 Looking-to-Listen-at-the-Cocktail-Party-master/data/audio_data/audio_data.py")

    print("Generating face embeddings...")
    run_command("python3 Looking-to-Listen-at-the-Cocktail-Party-master/data/Tensorflow_to_Keras.py")
    run_command("python3 Looking-to-Listen-at-the-Cocktail-Party-master/model/face_embedding/face_emb.py")

    print("Creating AVdataset_train.txt and AVdataset_val.txt...")
    run_command("python3 Looking-to-Listen-at-the-Cocktail-Party-master/data/AV_data_log.py")

def train_model():
    """Train the model."""
    print("Starting training...")
    run_command("python3 Looking-to-Listen-at-the-Cocktail-Party-master/train.py")

def test_model():
    """Test the model."""
    print("Starting testing...")
    run_command("python3 Looking-to-Listen-at-the-Cocktail-Party-master/test.py")

def update_benchmarks():
    print("Benchmark Update...")
    run_command("python3 benchmarks.py")

def main():
    """Main script to execute all steps."""
    print("Setting up the environment...")
    #setup_environment()

    #print("Downloading required datasets and files...")
    #download_data()

    print("Preprocessing data...")
    preprocess_data()

    print("Training the model...")
    train_model()

    print("Testing the model...")
    test_model()

    print("Update benchmarks...")
    test_model()

    print("All steps completed successfully!")

if __name__ == "__main__":
    main()
