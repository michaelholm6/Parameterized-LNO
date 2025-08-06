import Libraries.LNO as LNO
import torch
<<<<<<<< HEAD:Parameterized_LNO/main.py
from Parameterized_LNO import fd
========
import Libraries.functional_dataloaders as fd
>>>>>>>> parent of 77cecef (Completed code to combine LES SINDy and the LNO):My Code/main.py
import pandas as pd

LNO_training_dataset = pd.read_csv('Training Data\Duffing Oscillator\duffing_oscillator_training_data.csv')
#LNO_sparse_training_dataset = fd.SparsePandasDataset(LNO_training_dataset, .01, 0)
LNO_sparse_training_dataset = fd.PandasDataset(LNO_training_dataset)
LNO_training_dataloader = torch.utils.data.DataLoader(LNO_sparse_training_dataset, batch_size=32, shuffle=True)
LNO_validation_dataset = pd.read_csv('Training Data\Duffing Oscillator\duffing_oscillator_test_data.csv')
#LNO_sparse_validation_dataset = fd.SparsePandasDataset(LNO_validation_dataset, .1, 1)
LNO_validation_dataset = fd.PandasDataset(LNO_validation_dataset)
LNO_validation_dataloader = torch.utils.data.DataLoader(LNO_validation_dataset, batch_size=32, shuffle=True)

LNO_trainer = LNO.LNO_trainer(4, 16, LNO_training_dataloader, LNO_validation_dataloader)
LNO_trainer.train(1000, 1e-3, graphing_frequency=150)
LNO_trainer.graph_loss()
