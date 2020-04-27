import os, time, glob, shutil
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.pyplot import specgram
import librosa
import librosa.display
import numpy as np
from pathlib import Path 
from fastai import *
from fastai.vision import *
import pandas as pd

##### LOG #####
import logging
main_logger = logging.getLogger('gpg')
main_logger.setLevel(logging.DEBUG)
fname = 'gopigo.log' # Any name for the log file

# Create the FileHandler object. This is required!
fh = logging.FileHandler(fname, mode='w')
fh.setLevel(logging.INFO)  # Will write to the log file the messages with level >= logging.INFO

# The following row is strongly recommended for the GoPiGo Test!
fh_formatter = logging.Formatter('%(relativeCreated)d,%(name)s,%(message)s')
fh.setFormatter(fh_formatter)
main_logger.addHandler(fh)
    
# The StreamHandler is optional for you. Use it just to debug your program code
sh = logging.StreamHandler()
sh_formatter = logging.Formatter('%(relativeCreated)8d %(name)s %(levelname)s %(message)s')
sh.setLevel(logging.DEBUG)
sh.setFormatter(sh_formatter)
main_logger.addHandler(sh)
main_logger.debug('Logger started') # This debug message will be handled only by StreamHendler
##########

audio_path = r"C:\Users\sasha\OneDrive\Desktop\Robotics\listening\record"
spectrogram_path = Path(r'C:\Users\sasha\OneDrive\Desktop\Robotics\listening\spectrogram')
model_path = r"C:\Users\sasha\OneDrive\Desktop\Robotics\listening\models" 

model1 = load_learner(model_path,'model_1.pkl')
model2 = load_learner(model_path,'model_2.pkl')
model3 = load_learner(model_path,'model_3.pkl')
model4 = load_learner(model_path,'model_4.pkl')
model5 = load_learner(model_path,'model_5.pkl')
model6 = load_learner(model_path,'model_6.pkl')
model7 = load_learner(model_path,'model_7.pkl')
model8 = load_learner(model_path,'model_8.pkl')
model9 = load_learner(model_path,'model_9.pkl')
model10 = load_learner(model_path,'model_10.pkl') 

index_values = ['air_conditioner','car_horn','children_playing','dog_bark','drilling','engine_idling','gun_shot','jackhammer','siren','street_music']

if not os.path.exists(spectrogram_path):
    os.makedirs(spectrogram_path)

before = dict ([(f, None) for f in os.listdir (audio_path)])

mic_logger = logging.getLogger('gpg.mic')
mic_logger.info('Start')


try:
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
            S = librosa.feature.melspectrogram(y=samples, sr=8000)
            librosa.display.specshow(librosa.power_to_db(S, ref=np.max))
            plt.savefig(filename, dpi=400, bbox_inches='tight',pad_inches=0)
            plt.close('all')

            print('\n') 
            print('Spectrogram Created')

            image = filename

            model1probs = pd.Series(model1.predict(open_image(image))[2],index = index_values, name = 'model1')
            model2probs = pd.Series(model2.predict(open_image(image))[2],index = index_values, name = 'model2')
            model3probs = pd.Series(model3.predict(open_image(image))[2],index = index_values, name = 'model3')
            model4probs = pd.Series(model4.predict(open_image(image))[2],index = index_values, name = 'model4')
            model5probs = pd.Series(model5.predict(open_image(image))[2],index = index_values, name = 'model5')
            model6probs = pd.Series(model6.predict(open_image(image))[2],index = index_values, name = 'model6')
            model7probs = pd.Series(model7.predict(open_image(image))[2],index = index_values, name = 'model7')
            model8probs = pd.Series(model8.predict(open_image(image))[2],index = index_values, name = 'model8')
            model9probs = pd.Series(model9.predict(open_image(image))[2],index = index_values, name = 'model9')
            model10probs = pd.Series(model10.predict(open_image(image))[2],index = index_values, name = 'model10')

            combined = pd.concat([model1probs,model2probs,model3probs,model4probs,model5probs,model6probs,model7probs,model8probs,model9probs,model10probs],axis=1)
            
            
            class_name = combined.mean(axis=1).idxmax()
            mic_logger.info(class_name)
            
            print('Prediction:', class_name)

      if removed:
          print("Removed: ", ", ".join (removed))
      before = after
  
except KeyboardInterrupt:
    mic_logger.info('Finish')
    print('Program Ended')
  