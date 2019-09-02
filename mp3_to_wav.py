from pydub import AudioSegment
import glob
import os
path = '/home/lms/Documents/speech-music-classify-lms'
export_path = os.path.join(path, 'music/')

music_mp3_files = glob.glob(os.path.join(path, 'music', '*.mp3'))
i = 0
for mp3 in music_mp3_files:
    sound = AudioSegment.from_mp3(mp3)
    name = i
    sound.export(path+'/music/'+str(name)+'.wav',format ='wav')
    i += 1

music_wav_files = glob.glob(os.path.join(path, 'music', '*.wav'))
i = 0
for wav in music_wav_files:
    sound = AudioSegment.from_file(wav)
    name = i
    sound = sound.set_frame_rate(16000).set_channels(1)
    print(sound.frame_rate)
    sound.export(path+'/music/'+str(name)+'.wav',format ='wav')
    i += 1