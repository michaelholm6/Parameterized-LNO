if __name__ == '__main__':
    import sys
    import os
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from Parameterized_LNO.Libraries import functional_dataloaders as fd
from Parameterized_LNO.Libraries import LES_SINDy_LNO as les_sindy_lno
from Parameterized_LNO.Libraries import LES_Sindy as sindy
from Parameterized_LNO.Libraries import LNO as lno
import pandas as pd
import torch
import numpy as np

class LES_SINDy_LNO_test_harness:
    def __init__(self, training_data_path, sample_frequency, noise_std_dev, validation_data_path, lno_width, lno_poles, lno_epochs, lno_learning_rate, 
                 sindy_functions, sindy_derivatives, sindy_s_values, sindy_data_percent):
        training_dataset = pd.read_csv(training_data_path)
        sparse_training_dataset = fd.SparsePandasDataset(training_dataset, sample_frequency, noise_std_dev)
        training_dataloader = torch.utils.data.DataLoader(sparse_training_dataset, batch_size=32, shuffle=True)
        validation_dataset = pd.read_csv(validation_data_path)
        validation_dataset = fd.PandasDataset(validation_dataset)
        validation_dataloader = torch.utils.data.DataLoader(validation_dataset, batch_size=32, shuffle=True)
        LNO_model = lno.LNO1d(lno_width, lno_poles)
        self.LNO_trainer_object = lno.LNO_trainer(LNO_model, training_dataloader, validation_dataloader)
        sindy_model = sindy.LES_SINDy_model(sindy_functions, sindy_derivatives, sindy_s_values)
        self.LES_SINDy_trainer_object = sindy.LES_SINDy_trainer(sindy_model, training_dataloader, data_percentage=sindy_data_percent)
        
        sindy_lno_model = les_sindy_lno.les_sindy_lno(
            width=lno_width,
            poles=lno_poles,
            SINDy_functions=sindy_functions,
            derivatives=sindy_derivatives,
            s_values=sindy_s_values,
        )
        
        self.sindy_lno_trainer = les_sindy_lno.les_sindy_lno_trainer(
            model=sindy_lno_model,
            dataloader=training_dataloader,
            valiloader=validation_dataloader,
            sindy_data_percent=sindy_data_percent,
            LNO_training_epochs=lno_epochs,
            LNO_learning_rate=lno_learning_rate,
            les_sindy_trainer=self.LES_SINDy_trainer_object
        )
        
        self.lno_epochs = lno_epochs
        self.lno_learning_rate = lno_learning_rate

    def test(self): 
        
        self.LNO_trainer_object.train(epochs=self.lno_epochs, learning_rate=self.lno_learning_rate)
        self.LNO_trainer_object.graph_loss()
        self.LES_SINDy_trainer_object.train()
        self.sindy_lno_trainer.train()
        
        
        print(f"LES SINDy coefficients:{self.LES_SINDy_trainer_object.model.coefficients}")
        self.sindy_lno_trainer.graph_sindy_coefficients([1, .5, 1, 1])
        self.sindy_lno_trainer.graph_LNO_loss()
        
if __name__ == '__main__':
    
    x = lambda j : j

    x_cubed = lambda j : j**3
    
    test_harness = LES_SINDy_LNO_test_harness(
        training_data_path='Training Data/Duffing Oscillator/duffing_oscillator_training_data.csv',
        sample_frequency=0.05,
        noise_std_dev=0.006,
        validation_data_path='Training Data/Duffing Oscillator/duffing_oscillator_test_data.csv',
        lno_width=16,
        lno_poles=4,
        lno_epochs=5,
        lno_learning_rate=0.001,
        sindy_functions=[x, x, x, x_cubed],
        sindy_derivatives=[0, 1 ,2 , 0],
        sindy_s_values=np.arange(1, 3, .5),
        sindy_data_percent=1
    )
    
    test_harness.test()
        
        
        
    