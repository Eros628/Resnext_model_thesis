import os
from pydub import AudioSegment

# Folder containing FLAC files
input_folder = r"D:\Thesis traning\asvspoof5\flac_T"
output_folder =r"D:\Thesis traning\asvspoof5\wav_T"

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Loop through all files in input folder
for filename in os.listdir(input_folder):
    if filename.lower().endswith(".flac"):
        flac_path = os.path.join(input_folder, filename)
        wav_filename = os.path.splitext(filename)[0] + ".wav"
        wav_path = os.path.join(output_folder, wav_filename)

        print(f"Converting {filename} → {wav_filename}")
        sound = AudioSegment.from_file(flac_path, format="flac")
        sound.export(wav_path, format="wav")

print("✅ Conversion complete!")