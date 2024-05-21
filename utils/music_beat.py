import librosa

def get_beat(wave_path):
    # get the locations of the beats and tempo
    y, sr = librosa.load(wave_path)
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    beats_time = librosa.frames_to_time(beats, sr=sr)
    return beats_time

if __name__ == '__main__':
    wave_path = r"/data/jdx/code/m2beats_test/MOVIE_ASSET/music/believer.wav"
    beats = get_beat(wave_path)
    print(beats)