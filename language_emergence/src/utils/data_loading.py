import pickle
from torch.utils.data import Dataset

class QMatrixData(Dataset):
    def __init__(self, q_file_path, label_file_path):
        with open(q_file_path, 'rb') as q_file:
            self.q_matrices = pickle.load(q_file)
        with open(label_file_path, 'rb') as label_file:
            self.labels = pickle.load(label_file)

    def __len__(self):
        return len(self.q_matrices)
    
    def __getitem__(self, index):
        return self.q_matrices[index], self.labels[index]
