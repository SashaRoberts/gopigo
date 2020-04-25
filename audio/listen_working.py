#C:\\Users\\sasha\\OneDrive\\Desktop\\Robotics\\listening\\record

import pyaudio
import math
import struct
import wave
import time
import os

Threshold = 30

SHORT_NORMALIZE = (1.0/32768.0)
chunk = 1024*4
FORMAT = pyaudio.paInt16 #8
CHANNELS = 1
RATE = 44100
swidth = 2
record_secs = 2

#TIMEOUT_LENGTH = 2

f_name_directory = 'record'

class Recorder:

    @staticmethod
    def rms(frame):
        count = len(frame) / swidth
        format = "%dh" % (count)
        shorts = struct.unpack(format, frame)

        sum_squares = 0.0
        for sample in shorts:
            n = sample * SHORT_NORMALIZE
            sum_squares += n * n
        rms = math.pow(sum_squares / count, 0.5)

        return rms * 1000

    def __init__(self):
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(format=FORMAT,
                                  channels=CHANNELS,
                                  rate=RATE,
                                  input=True,
                                  output=True,
                                  frames_per_buffer=chunk)

    def record(self):

        print('Noise detected, recording beginning')
        rec = []

        for ii in range(0,int((RATE/chunk)*record_secs)):
            data = self.stream.read(chunk,exception_on_overflow = False)
            rec.append(data)
            
        self.write(b''.join(rec))

        print("finished recording")
        
        self.write(b''.join(rec)) #rec

    def write(self, recording):
        n_files = len(os.listdir(f_name_directory))

        filename = os.path.join(f_name_directory, '{}.wav'.format(n_files))

        wf = wave.open(filename, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(self.p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(recording)
        wf.close()
        print('Written to file: {}'.format(filename))
        print('Returning to listening')



    def listen(self):
        print('Listening beginning')
        while True:
            input = self.stream.read(chunk, exception_on_overflow=False)
            rms_val = self.rms(input)
            if rms_val > Threshold:
                self.record()

a = Recorder()

a.listen()