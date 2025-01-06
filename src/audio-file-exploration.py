
import pandas as pd
import numpy as np
import librosa
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.decomposition import PCA
import seaborn as sns



audio_data_path = 'data/audio'
plt_image_path = 'output/images/'
audio_file = None
y = None
sr = None
data = None

def audio_data_exploration():
    global y, sr
    y, sr = librosa.load(f'{audio_data_path}/genres_original/reggae/reggae.00036.wav')
    print('y:', y, '\n')
    print('y shape:', np.shape(y), '\n')
    print('Sample Rate (KHz):', sr, '\n')
    global audio_file 
    audio_file, _ = librosa.effects.trim(y)
    # the result is an numpy ndarray
    print('Audio File:', audio_file, '\n')
    print('Audio File shape:', np.shape(audio_file))
    plt.figure(figsize = (16, 6))
    librosa.display.waveshow(y = audio_file, sr = sr, color = "#A300F9");
    plt.title("Sound Waves in Reggae 36", fontsize = 23);
    plt.savefig(plt_image_path+"sound_waves_reggae36.png", dpi=300, bbox_inches="tight")
    
    # Default FFT window size
    n_fft = 2048 # FFT window size
    hop_length = 512 # number of audio frames between STFT columns (looks like a good default)

    # Short-time Fourier transform (STFT)
    D = np.abs(librosa.stft(audio_file, n_fft = n_fft, hop_length = hop_length))
    print('Shape of D object:', np.shape(D))
    plt.figure(figsize = (16, 6))
    plt.plot(D);
    plt.title("SFTF", fontsize = 23);
    plt.savefig(plt_image_path+"sftf.png", dpi=300, bbox_inches="tight")
    
    # Convert an amplitude spectrogram to Decibels-scaled spectrogram.
    DB = librosa.amplitude_to_db(D, ref = np.max)

    # Creating the Spectogram
    plt.figure(figsize = (16, 6))
    librosa.display.specshow(DB, sr = sr, hop_length = hop_length, x_axis = 'time', y_axis = 'log',
                            cmap = 'cool')
    plt.colorbar();
    plt.title("Spectrogram", fontsize = 23);
    plt.savefig(plt_image_path+"spectrogram.png", dpi=300, bbox_inches="tight")
    
    # Mel Spectrogram is the result of some non-linear transformation of the frequency scale.
    y, sr = librosa.load(f'{audio_data_path}/genres_original/metal/metal.00036.wav')
    y, _ = librosa.effects.trim(y)
    #librosa.feature.melspectrogram(y)

    S = librosa.feature.melspectrogram(y=y, sr=sr)
    S_DB = librosa.amplitude_to_db(S, ref=np.max)
    plt.figure(figsize = (16, 6))
    librosa.display.specshow(S_DB, sr=sr, hop_length=hop_length, x_axis = 'time', y_axis = 'log',
                            cmap = 'cool');
    plt.colorbar();
    plt.title("Metal Mel Spectrogram", fontsize = 23);
    plt.savefig(plt_image_path+"metal_mel_spectrogram.png", dpi=300, bbox_inches="tight")
    
    
    # Classical genres audio file analysis
    y, sr = librosa.load(f'{audio_data_path}/genres_original/classical/classical.00036.wav')
    y, _ = librosa.effects.trim(y)
    S = librosa.feature.melspectrogram(y=y, sr=sr)
    S_DB = librosa.amplitude_to_db(S, ref=np.max)
    plt.figure(figsize = (16, 6))
    librosa.display.specshow(S_DB, sr=sr, hop_length=hop_length, x_axis = 'time', y_axis = 'log',
                            cmap = 'cool');
    plt.colorbar();
    plt.title("Classical Mel Spectrogram", fontsize = 23);
    plt.savefig(plt_image_path+"classical_mel_spectrogram.png", dpi=300, bbox_inches="tight")
    plt.close()  

def zero_crossing_rate():
    print()
    # Total zero_crossings in our 1 song
    zero_crossings = librosa.zero_crossings(audio_file, pad=False)

    print(sum(zero_crossings))
    print(zero_crossings)
    print(np.mean(zero_crossings))
    y_harm, y_perc = librosa.effects.hpss(audio_file)

    # Harmonics and Perceptrual
    plt.figure(figsize = (16, 6))
    plt.plot(y_harm, color = '#A300F9');
    plt.plot(y_perc, color = '#FFB100');
    plt.title("Harmonics and Perceptrual", fontsize = 23);
    plt.savefig(plt_image_path+"harmonics_and_perceptrual.png", dpi=300, bbox_inches="tight")
    plt.close()  
    
    # Tempo BMP (beats per minute)¶
    # Dynamic programming beat tracker.
    
    tempo, _ = librosa.beat.beat_track(y=y, sr = sr)
    print(tempo)
    # Spectral Centroid
    # indicates where the ”centre of mass” for a sound is located and is calculated as the weighted mean of the frequencies present in the soun
    spectral_centroids = librosa.feature.spectral_centroid(y=audio_file, sr=sr)[0]
    # Shape is a vector
    print('Centroids:', spectral_centroids, '\n')
    print('Shape of Spectral Centroids:', spectral_centroids.shape, '\n')
    # Computing the time variable for visualization
    frames = range(len(spectral_centroids))
    # Converts frame counts to time (seconds)
    t = librosa.frames_to_time(frames)
    print('frames:', frames, '\n')
    print('t:', t)
    plt.figure(figsize = (16, 6))
    librosa.display.waveshow(audio_file, sr=sr, alpha=0.4, color = '#A300F9');
    plt.plot(t, preprocessing.minmax_scale(spectral_centroids, axis=0), color='#FFB100');
    plt.title("Spectral Centroid", fontsize = 23);
    plt.savefig(plt_image_path+"spectral_centroid.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    # Spectral Rolloff
    # is a measure of the shape of the signal. It represents the frequency below which a specified percentage of the total spectral energy, e.g. 85%, lies
    # Spectral RollOff Vector
    spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_file, sr=sr)[0]
    # The plot
    plt.figure(figsize = (16, 6))
    librosa.display.waveshow(audio_file, sr=sr, alpha=0.4, color = '#A300F9');
    plt.plot(t, preprocessing.minmax_scale(spectral_rolloff, axis=0), color='#FFB100');
    plt.title("Spectral Rolloff", fontsize = 23);
    plt.savefig(plt_image_path+"spectral_rolloff.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    # Mel-Frequency Cepstral Coefficients: MFCC
    # The Mel frequency cepstral coefficients (MFCCs) of a signal are a 
    # small set of features (usually about 10–20) which concisely describe 
    # the overall shape of a spectral envelope. It models the characteristics of the human voice.
    mfccs = librosa.feature.mfcc(y=audio_file, sr=sr)
    print('mfccs shape:', mfccs.shape)

    #Displaying  the MFCCs:
    plt.figure(figsize = (16, 6))
    librosa.display.specshow(mfccs, sr=sr, x_axis='time', cmap = 'cool');
    plt.title("Mel-Frequency Cepstral Coefficients", fontsize = 23);
    plt.savefig(plt_image_path+"mfcc.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    # Perform Feature Scaling
    mfccs = preprocessing.scale(mfccs, axis=1)
    print('Mean:', mfccs.mean(), '\n')
    print('Var:', mfccs.var())

    plt.figure(figsize = (16, 6))
    librosa.display.specshow(mfccs, sr=sr, x_axis='time', cmap = 'cool');
    plt.title("MFCC Scaled", fontsize = 23);
    plt.savefig(plt_image_path+"mfcc_scaled.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    # Chroma features are an interesting and powerful 
    # representation for music audio in which the entire 
    # spectrum is projected onto 12 bins representing the 12 distinct 
    # semitones (or chroma) of the musical octave.
    
    # Increase or decrease hop_length to change how granular you want your data to be
    hop_length = 5000

    # Chromogram
    chromagram = librosa.feature.chroma_stft(y=audio_file, sr=sr, hop_length=hop_length)
    print('Chromogram shape:', chromagram.shape)
    plt.figure(figsize=(16, 6))
    librosa.display.specshow(chromagram, x_axis='time', y_axis='chroma', hop_length=hop_length, cmap='coolwarm');
    plt.title("Chroma features", fontsize = 23);
    plt.savefig(plt_image_path+"chroma_features.png", dpi=300, bbox_inches="tight")
    plt.close()

#EDA is going to be performed on the features_30_sec.csv. 
# This file contains the mean and variance for each audio file fo the features analysed above.
# So, the table has a final of 1000 rows (10 genrex x 100 audio files) 
# and 60 features (dimensionalities).
def eda_30sec_audiofiles_data():
    global data
    data = pd.read_csv(f'{audio_data_path}/features_30_sec.csv')
    print(data.head())
    #Correlation Heatmap for feature means¶
    
    # Computing the Correlation Matrix
    spike_cols = [col for col in data.columns if 'mean' in col]
    corr = data[spike_cols].corr()
    
    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=np.bool_))
    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(16, 11));
    
    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(0, 25, as_cmap=True, s = 90, l = 45, n = 5)
    
    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
    plt.title('Correlation Heatmap (for the MEAN variables)', fontsize = 25)
    plt.xticks(fontsize = 10)
    plt.yticks(fontsize = 10);
    plt.savefig(plt_image_path+"corr_heatmap.jpg",dpi=300, bbox_inches="tight")
    plt.close()
    
    # Box Plot for Genres Distributions
    
    x = data[["label", "tempo"]]

    f, ax = plt.subplots(figsize=(16, 9));
    sns.boxplot(x = "label", y = "tempo", data = x, palette = 'husl');
    plt.title('BPM Boxplot for Genres', fontsize = 25)
    plt.xticks(fontsize = 14)
    plt.yticks(fontsize = 10);
    plt.xlabel("Genre", fontsize = 15)
    plt.ylabel("BPM", fontsize = 15)
    plt.savefig(plt_image_path+"BPM_Boxplot.jpg", dpi=300, bbox_inches="tight")
 
# Principal Component Analysis - to visualize possible groups of genres¶
# Normalization
# PCA
# The Scatter Plot 
def principal_component_analysis():
    global data
    data = data.iloc[0:, 1:]  # Selecting all rows but removing 1st column from dataset (ie length)
    y = data['label']
    X = data.loc[:, data.columns != 'label']
    
    #### NORMALIZE X ####
    cols = X.columns
    min_max_scaler = preprocessing.MinMaxScaler()
    np_scaled = min_max_scaler.fit_transform(X)
    X = pd.DataFrame(np_scaled, columns = cols)
    
    #### PCA 2 COMPONENTS ####
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(X)
    principalDf = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2'])

    # concatenate with target label
    finalDf = pd.concat([principalDf, y], axis = 1)

    print(pca.explained_variance_ratio_)
    plt.figure(figsize = (16, 9))
    sns.scatterplot(x = "principal component 1", y = "principal component 2", data = finalDf, hue = "label", alpha = 0.7,
                s = 100);
    plt.title('PCA on Genres', fontsize = 25)
    plt.xticks(fontsize = 14)
    plt.yticks(fontsize = 10);
    plt.xlabel("Principal Component 1", fontsize = 15)
    plt.ylabel("Principal Component 2", fontsize = 15)
    plt.savefig(plt_image_path+"PCA_Scattert.jpg", dpi=300, bbox_inches="tight")
 

##################### END OF METHOD ################## 
 
            
##### Main Entry Point #######
# Main code

if __name__ == "__main__":
    print("Audio file analysis initiated")
    audio_data_exploration()
    zero_crossing_rate()
    eda_30sec_audiofiles_data()
    principal_component_analysis()