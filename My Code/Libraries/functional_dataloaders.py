import torch
from torch.utils.data import Dataset
import numpy as np
import random

class PandasDataset(Dataset):
    def __init__(self, dataframe):
        dataframe = dataframe.replace({r'\n': '', r'\r\n': '', r'\r': ''}, regex=True)
        for column in dataframe.columns[1:]:
            dataframe[column] = dataframe[column].apply(lambda x: x.replace('[', '').replace(']', ''))
            dataframe[column] = dataframe[column].apply(lambda x: x.split())
            dataframe[column] = dataframe[column].apply(lambda x: [float(i) for i in x])

        self.dataframe = dataframe
        # Separate features and target
        self.features = np.array([dataframe['f_of_t'].values, dataframe['f_t'].values]).T
        self.labels = np.array([dataframe['x_of_t'].values, dataframe['x_t'].values]).T
        self.sample_frequency = dataframe['x_t'].iloc[0][1] - dataframe['x_t'].iloc[0][0]
    
    def __len__(self):
        # Return the number of samples in the dataset
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        # Return a specific sample (features, label) from the DataFrame
        feature = torch.cat((torch.tensor(self.features[idx][0], dtype=torch.float32).unsqueeze(-1), torch.tensor(self.features[idx][1], dtype=torch.float32).unsqueeze(-1)), dim=1)
        label = torch.cat((torch.tensor(self.labels[idx][0], dtype=torch.float32).unsqueeze(-1), torch.tensor(self.labels[idx][1], dtype=torch.float32).unsqueeze(-1)), dim=1)
        return feature, label
    
class SparsePandasDataset(PandasDataset):
    def __init__(self, dataframe, sample_frequency, noise_std_dev):
        super().__init__(dataframe)
        
        sample_steps = int(round(sample_frequency/self.sample_frequency))
        new_labels = np.ndarray(shape=(0, 2), dtype=object)
        
        
        for datapoint in range(self.features.shape[0]):   
            new_label_list = []
            time_list = []
            i = 0
            while i * sample_steps < len(self.features[datapoint][0]):
                time_list.append(self.features[datapoint][1][i * sample_steps])
                new_label_list.append(self.labels[datapoint][0][i * sample_steps] + random.gauss(0, noise_std_dev))
                for j in range(sample_steps - 1):
                    time_list.append(np.nan)
                    new_label_list.append(np.nan)
                i += 1
            new_labels = np.append(new_labels, np.array([[0, 0]]), axis=0)
            new_labels[-1][:] = [new_label_list[0: len(self.labels[0][0])], time_list[0: len(self.labels[0][0])]]
                             
        self.labels = new_labels
            
        