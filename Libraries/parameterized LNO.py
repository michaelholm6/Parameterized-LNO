# if __name__ == '__main__':
#     import sys
#     import os
#     sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# import torch
# from torch import nn
# import numpy as np
# import time
# from matplotlib import pyplot as plt
# import Parameterized_LNO.Libraries.functional_dataloaders as fd
# import pandas as pd
# import os
# import torch.nn.functional as F

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# class PR(nn.Module):
#     def __init__(self, in_channels, out_channels, number_of_kernel_poles, parameter_dim):
#         super(PR, self).__init__()

#         self.out_channels = out_channels
#         self.in_channels = in_channels
#         self.number_of_kernel_poles = number_of_kernel_poles

#         output_dim = in_channels * out_channels * number_of_kernel_poles

#         # MLP to generate real-valued poles
#         self.parameter_to_poles = nn.Sequential(
#             nn.Linear(parameter_dim, 256),
#             nn.ReLU(),
#             nn.Linear(256, output_dim)
#         )

#         # MLP to generate real-valued residues
#         self.parameter_to_residues = nn.Sequential(
#             nn.Linear(parameter_dim, 256),
#             nn.ReLU(),
#             nn.Linear(256, output_dim)
#         )

#     def reshape_weights(self, flat_weights, batch_size):
#         return flat_weights.view(batch_size, self.in_channels, self.out_channels, self.number_of_kernel_poles)

#     def output_PR(self, omega, alpha, weights_pole, weights_residue):
#         # omega: [freq], weights_pole/residue: [B, in, out, num_poles]
#         # B, in_ch, out_ch, num_poles = weights_pole.shape
#         freq = omega.shape[0]

#         # Reshape omega to broadcast: [1, freq, 1, 1, 1]
#         omega = omega.view(1, freq, 1, 1, 1)

#         # # Reshape poles and residues: [B, 1, in, out, num_poles]
#         weights_pole = weights_pole.unsqueeze(1)     # [B, 1, in, out, num_poles]
#         weights_residue = weights_residue.unsqueeze(1)

#         # K_of_s: [B, freq, in, out, num_poles]
#         K_of_s = weights_residue / (omega - weights_pole)

#        # lambda_l: [B, freq, out]
#         # We want to sum over the `in` and `num_poles` dimensions
#         gamma_n = torch.einsum("bxi,bxiok->biok", alpha, -K_of_s)

#         # gamma_n: [B, freq, in, out, num_poles]
#         lambda_l = torch.einsum("bxi,bxiok->box", alpha, K_of_s)

#         return lambda_l, gamma_n

#     def forward(self, x, times, parameters):
#         batch_size = x.shape[0]
#         seq_len = times.shape[1]

#         dt = (times[0, 1] - times[0, 0]).item()
#         alpha_l = torch.fft.fft(x)
#         omega_l_times_i = torch.fft.fftfreq(times.shape[1], dt)*2*np.pi*1j #equation 10

#         omega_l_times_i_expanded = omega_l_times_i
        
#         # for i in range(3):
#         #     omega_l_times_i_expanded = omega_l_times_i_expanded.unsqueeze(-1)

#         parameters = parameters.view(batch_size, -1)
        
#         # Project parameters into poles and residues
#         poles_raw = self.parameter_to_poles(parameters)
#         residues_raw = self.parameter_to_residues(parameters)

#         weights_pole = self.reshape_weights(poles_raw, batch_size)  # [B, in, out, num_poles]
#         weights_residue = self.reshape_weights(residues_raw, batch_size)  # [B, in, out, num_poles]

#         # Expand alpha to match einsum dimensions: [B, freq, in]
#         lambda_l, gamma_n = self.output_PR(omega_l_times_i_expanded, alpha_l, weights_pole, weights_residue)

#         # Reconstruct time-domain outputs
#         x1 = torch.fft.ifft(lambda_l, n=seq_len)  # inverse real FFT
#         x1 = torch.real(x1) #second term of equation 18 (because this is the same as the inverse dft)  
#         weights_pole_times_t = torch.einsum("biok,t->biokt", weights_pole, times[0].type(torch.complex64).reshape(-1)) #Assumes all sample times are the same
#         weights_pole_times_t_exponent = torch.exp(weights_pole_times_t) 
#         x2=torch.einsum("biok,biokt->bot", gamma_n, weights_pole_times_t_exponent) 
#         x2=torch.real(x2) #first term of equation 18
#         #x2=x2/x.size(-2) I dont think this is needed
        
#         return x1+x2 #equation 18
    
# class LNO1d(nn.Module):
#     def __init__(self, width, number_of_kernel_poles, parameter_dim=2):
#         super(LNO1d, self).__init__()

#         self.width = width
#         self.number_of_kernel_poles = number_of_kernel_poles
#         self.fc0 = nn.Linear(1, self.width) 
#         self.parameter_dim = parameter_dim

#         self.conv0 = PR(self.width, self.width, self.number_of_kernel_poles, self.parameter_dim)

#         # This replaces the learnable conv1d layer `self.w0`
#         self.param_to_conv_weight = nn.Sequential(
#             nn.Linear(parameter_dim, 256),
#             nn.ReLU(),
#             nn.Linear(256, width * width)  # kernel_size is 1, so only width x width
#         )

#         self.fc1 = nn.Linear(self.width, 128)
#         self.fc2 = nn.Linear(128, 1)

#     def forward(self, x: torch.Tensor, parameters: torch.Tensor):
#         """
#         parameters: shape [batch_size, parameter_dim]
#         """
#         times = x[:, :, 1]
#         x = x[:, :, 0]
#         x = x.unsqueeze(-1)  # shape: [B, N, 1]

#         x = self.fc0(x)  # shape: [B, N, width]

#         x1 = self.conv0(x, times, parameters)

#         # Project parameters into a conv1d weight tensor
#         while parameters.dim() < 2:
#             parameters = parameters.unsqueeze(-1)
        
#         conv_weight = self.param_to_conv_weight(parameters)  # shape: [B, width * width]
#         conv_weight = conv_weight.view(-1, self.width, self.width, 1)  # shape: [B, out, in, 1]

#         x_permuted = x.permute(0, 2, 1)  # [B, width, N]
#         x2 = torch.stack([
#             F.conv1d(x_permuted[i:i+1], conv_weight[i], bias=None, padding=0)
#             for i in range(x.shape[0])
#         ], dim=0).squeeze(1)  # [B, width, N]

#         x = x1 + x2 # [B, N, width]
        
#         x = x.permute(0, 2, 1)

#         x = self.fc1(x)
#         x = torch.sin(x)
#         x = self.fc2(x)
#         x = x.squeeze(-1)
        
#         return x
    
# class NormalizedMSELoss(nn.Module):
#     def __init__(self):
#         super(NormalizedMSELoss, self).__init__()

#     def forward(self, y_pred, y_true):
        
#         # Normalized MSE loss (only using valid (non-NaN) values)
#         loss = torch.mean((y_pred - y_true)**2) / torch.mean(y_true**2)
#         return loss
    
#     def __call__(self, y_pred, y_true):
#         return self.forward(y_pred, y_true)
    
    
# class LNO_trainer():
#     def __init__(self, LNO_model, dataloader, valiloader):
#         self.model = LNO_model
#         self.dataloader = dataloader
#         self.valiloader = valiloader
        
#     def train(self, epochs, learning_rate, graphing_frequency=0):
#         optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=0)
#         loss_fn = NormalizedMSELoss()
#         start_time = time.time()
#         lowest_validation = np.inf
        
#         self.train_loss = np.zeros(epochs)
#         self.vali_loss = np.zeros(epochs)
        
#         for epoch in range(epochs):
#             self.model.train()
#             epoch_loss = 0
#             for batch, (x, y, parameters) in enumerate(self.dataloader):
#                 x = x.to(device)
#                 y = y[:, :, 0].to(device)
#                 parameters = parameters.to(device)
#                 optimizer.zero_grad()
#                 y_pred = self.model(x, parameters)
#                 loss = loss_fn(y_pred, y) #* (1 - sindy_data_loss_weight)  # Apply weight to loss
                
#                 epoch_loss += loss.item() #* (1 - sindy_data_loss_weight)  # Apply weight to epoch loss  
#                 loss.backward()
#                 optimizer.step()
        
#             self.train_loss[epoch] = epoch_loss / len(self.dataloader)
            
#             self.model.eval()
            
#             with torch.no_grad():
#                 vali_epoch_loss = 0
#                 vali_epoch_error = 0
#                 for batch, (x, y, parameters) in enumerate(self.valiloader):
#                     x = x.to(device)
#                     y = y[:, :, 0].to(device)
#                     parameters = parameters.to(device)
#                     y_pred = self.model(x, parameters)
#                     loss = loss_fn(y_pred, y)
#                     vali_epoch_loss += loss.item()
                    
#                 self.vali_loss[epoch] = vali_epoch_loss / len(self.valiloader)
                
#             if self.vali_loss[epoch] < lowest_validation:
#                 lowest_validation = self.vali_loss[epoch]
#                 torch.save(self.model.state_dict(), 'LNO_model.pth')
#                 print(f"New best model saved at epoch {epoch+1} with validation loss: {lowest_validation:.4f}")
    
#             print(f"Epoch {epoch+1}/{epochs} - Loss: {self.train_loss[epoch]:.4f} - Vali Loss: {self.vali_loss[epoch]:.4f} - Time: {time.time()-start_time:.2f}s")
            
#             if graphing_frequency > 0 and epoch % graphing_frequency == 0:
#                 validation_times = self.valiloader.dataset[3][0][:, 1].numpy()
#                 y_of_t_validation = self.valiloader.dataset[3][1][:, 0].numpy()
#                 f_of_t_validation = self.valiloader.dataset[3][0][:, 0].numpy()
#                 y_pred_validation = self.model(self.valiloader.dataset[3][0].unsqueeze(0), self.valiloader.dataset[3][2]).detach().numpy()
                
#                 test_times = self.dataloader.dataset[3][0][:, 1].numpy()
#                 y_of_t_test = self.dataloader.dataset[3][1][:, 0].numpy()
#                 f_of_t_test = self.dataloader.dataset[3][0][:, 0].numpy()
#                 y_pred_test = self.model(self.dataloader.dataset[3][0].unsqueeze(0), self.dataloader.dataset[3][2]).detach().numpy()
                
#                 valid_mask = ~np.isnan(y_of_t_validation)
#                 valid_times = validation_times[valid_mask]
#                 valid_y = y_of_t_validation[valid_mask]
                
#                 plt.plot(valid_times, valid_y, label='True')
#                 plt.plot(validation_times, y_pred_validation[0], label='Predicted')
#                 plt.plot(validation_times, f_of_t_validation, label='Force')
#                 plt.legend()
#                 plt.title('Validation')
#                 plt.show()
                
#                 valid_mask = ~np.isnan(y_of_t_test)
#                 valid_times = test_times[valid_mask]
#                 valid_y = y_of_t_test[valid_mask]
                
#                 plt.plot(valid_times, valid_y, 'o', label='True', ms=1)
#                 plt.plot(test_times, y_pred_test[0], label='Predicted')
#                 plt.plot(test_times, f_of_t_test, label='Force')
#                 plt.legend()
#                 plt.title('Training')
#                 plt.show()
            
#     def graph_loss(self):
#         plt.plot(self.train_loss, label='train')
#         plt.plot(self.vali_loss, label='vali')
#         plt.legend()
#         plt.show()
        
# if __name__ == '__main__':
#     LNO_training_dataset = pd.read_csv(r'Training Data\Duffing Oscillator\duffing_oscillator_training_data.csv')
#     LNO_training_dataset = fd.PandasDataset(LNO_training_dataset)
#     LNO_training_dataloader = torch.utils.data.DataLoader(LNO_training_dataset, batch_size=32, shuffle=True)
#     LNO_validation_dataset = pd.read_csv(r'Training Data\Duffing Oscillator\duffing_oscillator_test_data.csv')
#     LNO_validation_dataset = fd.PandasDataset(LNO_validation_dataset)
#     LNO_validation_dataloader = torch.utils.data.DataLoader(LNO_validation_dataset, batch_size=32, shuffle=True)

#     LNO_model = LNO1d(4, 16, 1)
#     LNO_trainer_object = LNO_trainer(LNO_model, LNO_training_dataloader, LNO_validation_dataloader)
#     LNO_trainer_object.train(1000, 1e-3, graphing_frequency=10)
#     LNO_trainer_object.graph_loss()

        
import torch
from torch import nn
import numpy as np
import time
from matplotlib import pyplot as plt
import Parameterized_functional_dataloaders as pfd
import pandas as pd
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PR(nn.Module):
    def __init__(self, in_channels, out_channels, number_of_kernel_poles, parameter_dim):
        super(PR, self).__init__()
        
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.number_of_kernel_poles = number_of_kernel_poles
        
        output_dim = in_channels * out_channels * number_of_kernel_poles

        # MLP to generate real-valued poles
        self.parameter_to_poles = nn.Sequential(
        nn.Linear(parameter_dim, 64),
        nn.ReLU(),
        nn.Linear(64, output_dim),
        nn.Tanh()
    )
        
        self.scale = (1 / (in_channels*out_channels))
        self.weights_residue = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.number_of_kernel_poles, dtype=torch.cfloat, device=device))
       
    def output_PR(self, omega, alpha, weights_pole, weights_residue): 
        omega = omega.unsqueeze(0) # [1, freq, 1, 1, 1]  
        weights_pole = weights_pole.unsqueeze(1)     # [B, 1, in, out, num_poles]
        K_of_s = weights_residue*torch.div(1, torch.sub(omega ,weights_pole)) #Equation 17
        lambda_l=torch.einsum("bxi,bxiok->box", alpha, K_of_s) #Equation 16
        gamma_n=torch.einsum("bxi,bxiok->biok", alpha, -K_of_s) #Equation 14
        return lambda_l, gamma_n    

    def forward(self, x, times, parameters):
        #TODO: Determine dt and number of time steps
        dt = (times[0, 1] - times[0, 0]).item()
        alpha_l = torch.fft.fft(x)
        omega_l_times_i = torch.fft.fftfreq(times.shape[1], dt, device=times.device) * 2 * np.pi * 1j #equation 10
        
        omega_l_times_i_expanded = omega_l_times_i
        for i in range(3):
            omega_l_times_i_expanded = omega_l_times_i_expanded.unsqueeze(-1) #scaled up in dimensions to match width of LNO
    
        parameters = parameters.view(x.shape[0], -1)
        # Project parameters into poles
        poles_raw = self.parameter_to_poles(parameters)
        weights_pole = poles_raw.view(x.shape[0], self.in_channels, self.out_channels, self.number_of_kernel_poles)  # [B, in, out, num_poles]
    
        # Obtain output poles and residues for transient part and steady-state part
        lambda_l, gamma_n = self.output_PR(omega_l_times_i_expanded, alpha_l, weights_pole, self.weights_residue)
    
        # Obtain time histories of transient response and steady-state response
        x1 = torch.fft.ifft(lambda_l, n=times.size(-1))
        x1 = torch.real(x1) #second term of equation 18 (because this is the same as the inverse dft)  
        times_complex = times[0].to(torch.complex64)
        weights_pole_times_t = torch.einsum("biox,kz->bioxz", weights_pole, times_complex.reshape(1, -1)) #assumes all sample times are the same
        weights_pole_times_t_exponent = torch.exp(weights_pole_times_t) 
        x2=torch.einsum("biok,biokt->bot", gamma_n, weights_pole_times_t_exponent) 
        x2=torch.real(x2) #first term of equation 18
        #x2=x2/x.size(-2) I dont think this is needed
        return x1+x2 #equation 18

class LNO1d(nn.Module):
    def __init__(self, width, number_of_kernel_poles, parameter_dim):
        super(LNO1d, self).__init__()

        self.width = width
        self.number_of_kernel_poles = number_of_kernel_poles
        self.fc0 = nn.Linear(1, self.width) 

        self.conv0 = PR(self.width, self.width, self.number_of_kernel_poles, parameter_dim)
        self.w0 = nn.Conv1d(self.width, self.width, 1)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self,x, parameters):
        times = x[:, :, 1]
        x = x[:, :, 0]
        
        x = x.unsqueeze(-1)
        
        x = self.fc0(x)

        x1 = self.conv0(x, times, parameters)
        
        x = x.permute(0,2,1)
        x2 = self.w0(x)
        
        x = x1 + x2

        x = x.permute(0,2,1)

        x = self.fc1(x)
        x =  torch.sin(x) #non-linearity
        x = self.fc2(x)
        x = x.squeeze(-1)
        
        return x
    
class NormalizedMSELoss(nn.Module):
    def __init__(self):
        super(NormalizedMSELoss, self).__init__()

    def forward(self, y_pred, y_true):
        # Normalized MSE loss
        mask = ~torch.isnan(y_true)
        y_pred_valid = y_pred[mask]
        y_true_valid = y_true[mask]
        
        # Normalized MSE loss (only using valid (non-NaN) values)
        loss = torch.mean((y_pred_valid - y_true_valid)**2) / torch.mean(y_true_valid**2)
        return loss
    
    def __call__(self, y_pred, y_true):
        return self.forward(y_pred, y_true)
    
    
class LNO_trainer():
    def __init__(self, LNO_width, LNO_poles, parameter_dim, dataloader, valiloader):
        self.model = LNO1d(LNO_width, LNO_poles, parameter_dim).to(device)
        self.dataloader = dataloader
        self.valiloader = valiloader

        
    def train(self, epochs, learning_rate, graphing_frequency=0):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-4)
        loss_fn = NormalizedMSELoss()
        start_time = time.time()
        
        self.train_loss = np.zeros(epochs)
        self.vali_loss = np.zeros(epochs)
        
        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0
            epoch_error = 0
            for batch, (x, y, parameters) in enumerate(self.dataloader):
                x = x.to(device)
                y = y[:, :, 0].to(device)
                parameters = parameters.to(device)
                y_pred = self.model(x, parameters)
                loss = loss_fn(y_pred, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
        
            self.train_loss[epoch] = epoch_loss / len(self.dataloader)
            
            self.model.eval()
            
            with torch.no_grad():
                vali_epoch_loss = 0
                vali_epoch_error = 0
                for batch, (x, y, parameters) in enumerate(self.valiloader):
                    x = x.to(device)
                    y = y[:, :, 0].to(device)
                    parameters = parameters.to(device)
                    y_pred = self.model(x, parameters)
                    loss = loss_fn(y_pred, y)
                    vali_epoch_loss += loss.item()
                    
                self.vali_loss[epoch] = vali_epoch_loss / len(self.valiloader)
    
            print(f"Epoch {epoch+1}/{epochs} - Loss: {self.train_loss[epoch]:.4f} - Vali Loss: {self.vali_loss[epoch]:.4f} - Time: {time.time()-start_time:.2f}s")
            
            if graphing_frequency > 0 and epoch % graphing_frequency == 0:
                validation_times = self.valiloader.dataset[15][0][:, 1].numpy()
                y_of_t_validation = self.valiloader.dataset[15][1][:, 0].numpy()
                f_of_t_validation = self.valiloader.dataset[15][0][:, 0].numpy()
                y_pred_validation = self.model(self.valiloader.dataset[15][0].unsqueeze(0).to(device), self.valiloader.dataset[15][2].unsqueeze(0).to(device)).detach().cpu().numpy()
                
                test_times = self.dataloader.dataset[15][0][:, 1].numpy()
                y_of_t_test = self.dataloader.dataset[15][1][:, 0].numpy()
                f_of_t_test = self.dataloader.dataset[15][0][:, 0].numpy()
                y_pred_test = self.model(self.dataloader.dataset[15][0].unsqueeze(0).to(device), self.dataloader.dataset[15][2].unsqueeze(0).to(device)).detach().cpu().numpy()
                
                valid_mask = ~np.isnan(y_of_t_validation)
                valid_times = validation_times[valid_mask]
                valid_y = y_of_t_validation[valid_mask]
                
                plt.plot(valid_times, valid_y, label='True')
                plt.plot(validation_times, y_pred_validation[0], label='Predicted')
                plt.plot(validation_times, f_of_t_validation, label='Force')
                plt.legend()
                plt.title('Validation')
                plt.savefig('LNO_validation_plot.png')
                
                plt.clf()  # Clear the figure for the next plot
                
                valid_mask = ~np.isnan(y_of_t_test)
                valid_times = test_times[valid_mask]
                valid_y = y_of_t_test[valid_mask]
                
                plt.plot(valid_times, valid_y, label='True')
                plt.plot(test_times, y_pred_test[0], label='Predicted')
                plt.plot(test_times, f_of_t_test, label='Force')
                plt.legend()
                plt.title('Training')
                plt.savefig('LNO_training_plot.png')
                
                plt.clf()  # Clear the figure for the next plot
                
            
    def graph_loss(self):
        plt.plot(self.train_loss, label='train')
        plt.plot(self.vali_loss, label='vali')
        plt.legend()
        plt.savefig('LNO_loss_plot.png')
        plt.clf()  # Clear the figure after saving
        
if __name__ == '__main__':
    print(os.getcwd())
    file_path = os.path.join('Training Data', 'Duffing Oscillator', 'parameterized_duffing_oscillator_training_data.csv')
    LNO_training_dataset = pd.read_csv(file_path)
    #LNO_sparse_training_dataset = fd.SparsePandasDataset(LNO_training_dataset, .01, 0)
    LNO_sparse_training_dataset = pfd.PandasDataset(LNO_training_dataset)
    LNO_training_dataloader = torch.utils.data.DataLoader(LNO_sparse_training_dataset, batch_size=16, shuffle=True)
    file_path = os.path.join('Training Data', 'Duffing Oscillator', 'parameterized_duffing_oscillator_test_data.csv')
    LNO_validation_dataset = pd.read_csv(file_path)
    #LNO_sparse_validation_dataset = fd.SparsePandasDataset(LNO_validation_dataset, .1, 1)
    LNO_validation_dataset = pfd.PandasDataset(LNO_validation_dataset)
    LNO_validation_dataloader = torch.utils.data.DataLoader(LNO_validation_dataset, batch_size=16, shuffle=True)

    LNO_trainer_object = LNO_trainer(4, 16, 1, LNO_training_dataloader, LNO_validation_dataloader)
    LNO_trainer_object.train(1000, 1e-3, graphing_frequency=150)
    LNO_trainer_object.graph_loss()

        
        
        