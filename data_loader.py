import os
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd

class aistDataset(Dataset):
    """Custom data.Dataset compatible with data.DataLoader."""
    def __init__(self, movie_list_file, note_file_dir):
        self.movie_names = pd.read_csv(movie_list_file)
        self.note_file_dir = note_file_dir
        # self.keypoint_file_dir = keypoint_file_dir

    def __getitem__(self, idx):
        """Returns one data pair (source and target)."""
        movie_name = str(self.movie_names.iloc[idx, 0])
        note_file = os.path.join(self.note_file_dir, movie_name + '.npz')
        fr = np.load(note_file, allow_pickle=True)
        keypoints = fr['keypoints'][:1024] # 5000,17,3 
        timestamps = fr['timestamps'][:1024]
        beat = fr['beat'][:1024] # 5000,1   beat
        mask = fr['mask'][:1024]   # 5000,1

        # beat_y = np.zeros_like(beat)

        return keypoints, timestamps, beat, mask
    
    def __len__(self):
        return len(self.movie_names)


def get_dataloaders(train_data_file, note_file_dir, batch_size):
    # train_data_file = "/data/jdx/code/mycode/movie_list.csv"
    # note_file_dir = "/data/jdx/code/mycode/train_data"
    train_data = aistDataset(train_data_file, note_file_dir)
    dataset = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
    return dataset


if __name__ == "__main__":
    get_dataloaders()
