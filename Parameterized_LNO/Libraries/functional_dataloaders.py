import torch
from torch.utils.data import Dataset
import numpy as np

class PandasDataset(Dataset):
    def __init__(self, dataframe):
        # Clean newlines
        dataframe = dataframe.replace({r'\n': '', r'\r\n': '', r'\r': ''}, regex=True)

        # Parse list-like string columns into float arrays
        for column in dataframe.columns[1:]:
            if dataframe[column].apply(type).eq(str).all():
                dataframe[column] = dataframe[column].apply(lambda x: x.replace('[', '').replace(']', ''))
                dataframe[column] = dataframe[column].apply(lambda x: x.split())
                dataframe[column] = dataframe[column].apply(lambda x: [float(i) for i in x])

        self.dataframe = dataframe

        # Convert relevant columns to numpy arrays
        self.f_of_t = np.array(dataframe['f_of_t'].values)
        self.f_t = np.array(dataframe['f_t'].values)
        self.parameters = np.array(dataframe['parameters'].values)

        self.labels = np.array([dataframe['x_of_t'].values, dataframe['x_t'].values]).T
        self.sample_frequency = dataframe['x_t'].iloc[0][1] - dataframe['x_t'].iloc[0][0]

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        # Shape: [seq_len, 1]
        f_t_tensor = torch.tensor(self.f_t[idx], dtype=torch.float32).unsqueeze(-1)
        f_of_t_tensor = torch.tensor(self.f_of_t[idx], dtype=torch.float32).unsqueeze(-1)

        # Shape: [seq_len, 2]
        feature = torch.cat((f_of_t_tensor, f_t_tensor), dim=1)

        # Shape: [param_dim]
        parameter_tensor = torch.tensor(self.parameters[idx], dtype=torch.float32)

        # Shape: [seq_len, 1] each
        x_of_t_tensor = torch.tensor(self.labels[idx][0], dtype=torch.float32).unsqueeze(-1)
        x_t_tensor = torch.tensor(self.labels[idx][1], dtype=torch.float32).unsqueeze(-1)
        label = torch.cat((x_of_t_tensor, x_t_tensor), dim=1)

        return feature, label, parameter_tensor