import torch
import pandas as pd
from torch.utils.data import Dataset

column_names = ['X', 'Y', 'Z', 'Pressure', 'GripAngle', 'Timestamp', 'Test ID']
path_to_control = 'Process of creating/Dataset/control'
path_to_parkinson = 'Process of creating/Dataset/parkinson'

# Класс для преобразования датафрейма в Датасет из последовательностей и разметки данных
class SequenceDataset(Dataset):
    def __init__(self, data, labels=None, seq_len=200):
        self.seq_len = seq_len
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data) // self.seq_len

    def __getitem__(self, idx):
        start_idx = idx * self.seq_len
        end_idx = start_idx + self.seq_len
        x = self.data[start_idx:end_idx]

        if self.labels is not None:
            y = self.labels[idx]
            return torch.tensor(x, dtype=torch.float), torch.tensor(y, dtype=torch.float)
        else:
            return torch.tensor(x, dtype=torch.float)

# Функция для преобразования датафрейма в Датасет из последовательностей временных шагов
def prepare_data(dataframe, seq_len=200):
    features = dataframe[['X', 'Y', 'Z', 'Pressure', 'GripAngle']].values
    labels = dataframe['Label'].values  # Предполагаю, что Label - это целевые значения
    dataset = SequenceDataset(features, labels, seq_len)
    return dataset

# Функция для загрузки данных из одного файла – на выходе имеем Датафрэйм
def load_data(path_to_file):
    df = pd.read_csv(path_to_file, sep=';', names=column_names)
    df.drop('Timestamp', axis=1, inplace=True)
    df.drop('Test ID', axis=1, inplace=True)
    return df