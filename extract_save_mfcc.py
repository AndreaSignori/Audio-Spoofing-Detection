import librosa
import os
import pandas as pd
import numpy as np
#This program copmutes 20 MFCC for all files in given folder, then it saves it to csv file

def MFCC_extraction(audio_file):
    x, sample_rate = librosa.load(audio_file, res_type="soxr_hq")
    mfcc = np.mean(librosa.feature.mfcc(y=x, sr=sample_rate, n_mfcc=20).T, axis=0)

    return mfcc


directory_path = "ASVspoof2019_LA_train/flac/"
output_path = "mfcc_LA_train_features.csv"

mfcc_features_list = []

for file in os.listdir(directory_path):
    print(f"Processing file {file}")
    file_path = os.path.join(directory_path, file)
    mfcc_features = MFCC_extraction(file_path)
    mfcc_features_list.append([file] + list(mfcc_features))

df = pd.DataFrame(mfcc_features_list, columns=['filename'] + [f'mfcc_{i+1}' for i in range(20)])
df.to_csv(output_path, index=False)

print(f"MFCC features saved to {output_path}")
