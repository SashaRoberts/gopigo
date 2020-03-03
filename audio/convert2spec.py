import os, time, glob, shutil
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.pyplot import specgram
import librosa
import librosa.display
import numpy as np
from pathlib import Path  

audio_path = r"C:\Users\sasha\OneDrive\Desktop\Robotics\listening\record"
spectrogram_path = Path(r'C:\Users\sasha\OneDrive\Desktop\Robotics\listening\spectrogram')  

if not os.path.exists(spectrogram_path):
    os.makedirs(spectrogram_path)

before = dict ([(f, None) for f in os.listdir (audio_path)])

while 1:
  time.sleep (1)
  after = dict ([(f, None) for f in os.listdir (audio_path)])
  added = [f for f in after if not f in before]
  removed = [f for f in before if not f in after]
  if added:
      print("Added: ", ", ".join (added))
     
      #convert files to spectrogram 
      for i in tqdm(range(len(added))):
        file=added[i]
        audio_file=audio_path+"\\"+file
        
        samples, sample_rate = librosa.load(audio_file)
        fig = plt.figure(figsize=[0.72,0.72])
        ax = fig.add_subplot(111)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        ax.set_frame_on(False)
        filename  = spectrogram_path/Path(audio_file).name.replace('.wav','.png')
        S = librosa.feature.melspectrogram(y=samples, sr=sample_rate)
        librosa.display.specshow(librosa.power_to_db(S, ref=np.max))
        plt.savefig(filename, dpi=400, bbox_inches='tight',pad_inches=0)
        plt.close('all')

        print('spectrogram created')

  if removed:
      print("Removed: ", ", ".join (removed))
  before = after
