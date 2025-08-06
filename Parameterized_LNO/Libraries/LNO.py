import torch
from torch import nn
import numpy as np
import time
from matplotlib import pyplot as plt
import functional_dataloaders as fd
import pandas as pd



class PR(nn.Module):
    def __init__(self, in_channels, out_channels, number_of_kernel_poles):
        super(PR, self).__init__()
        
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.number_of_kernel_poles = number_of_kernel_poles
        self.scale = (1 / (in_channels*out_channels))
        self.weights_pole = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.number_of_kernel_poles, dtype=torch.cfloat))
        self.weights_residue = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.number_of_kernel_poles, dtype=torch.cfloat))
       
    def output_PR(self, omega, alpha, weights_pole, weights_residue):   
        K_of_s = weights_residue*torch.div(1, torch.sub(omega ,weights_pole)) #Equation 17
        lambda_l=torch.einsum("bxi,xiok->box", alpha, K_of_s) #Equation 16
        gamma_n=torch.einsum("bxi,xiok->biok", alpha, -K_of_s) #Equation 14
        return lambda_l, gamma_n    

    def forward(self, x, times):
        #TODO: Determine dt and number of time steps
        dt = (times[0, 1] - times[0, 0]).item()
        alpha_l = torch.fft.fft(x)
        omega_l_times_i = torch.fft.fftfreq(times.shape[1], dt)*2*np.pi*1j #equation 10
        
        omega_l_times_i_expanded = omega_l_times_i
        for i in range(3):
            omega_l_times_i_expanded = omega_l_times_i_expanded.unsqueeze(-1) #scaled up in dimensions to match width of LNO
    
        # Obtain output poles and residues for transient part and steady-state part
        lambda_l, gamma_n = self.output_PR(omega_l_times_i_expanded, alpha_l, self.weights_pole, self.weights_residue)
    
        # Obtain time histories of transient response and steady-state response
        x1 = torch.fft.ifft(lambda_l, n=times.size(-1))
        x1 = torch.real(x1) #second term of equation 18 (because this is the same as the inverse dft)  
        weights_pole_times_t = torch.einsum("iox,kz->ioxz", self.weights_pole, times[0].type(torch.complex64).reshape(1,-1)) #Assumes all sample times are the same
        weights_pole_times_t_exponent = torch.exp(weights_pole_times_t) 
        x2=torch.einsum("biok,iokt->bot", gamma_n, weights_pole_times_t_exponent) 
        x2=torch.real(x2) #first term of equation 18
        #x2=x2/x.size(-2) I dont think this is needed
        return x1+x2 #equation 18

class LNO1d(nn.Module):
    def __init__(self, width, number_of_kernel_poles):
        super(LNO1d, self).__init__()

        self.width = width
        self.number_of_kernel_poles = number_of_kernel_poles
        self.fc0 = nn.Linear(1, self.width) 

        self.conv0 = PR(self.width, self.width, self.number_of_kernel_poles)
        self.w0 = nn.Conv1d(self.width, self.width, 1)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self,x):
        times = x[:, :, 1]
        x = x[:, :, 0]
        
        x = x.unsqueeze(-1)
        
        x = self.fc0(x)

        x1 = self.conv0(x, times)
        
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
    def __init__(self, LNO_width, LNO_poles, dataloader, valiloader):
        self.model = LNO1d(LNO_width, LNO_poles)
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
            for batch, (x, y) in enumerate(self.dataloader):
                y = y[:, :, 0]
                optimizer.zero_grad()
                y_pred = self.model(x)
                loss = loss_fn(y_pred, y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
        
            self.train_loss[epoch] = epoch_loss / len(self.dataloader)
            
            self.model.eval()
            
            with torch.no_grad():
                vali_epoch_loss = 0
                vali_epoch_error = 0
                for batch, (x, y) in enumerate(self.valiloader):
                    y = y[:, :, 0]
                    y_pred = self.model(x)
                    loss = loss_fn(y_pred, y)
                    vali_epoch_loss += loss.item()
                    
                self.vali_loss[epoch] = vali_epoch_loss / len(self.valiloader)
    
            print(f"Epoch {epoch+1}/{epochs} - Loss: {self.train_loss[epoch]:.4f} - Vali Loss: {self.vali_loss[epoch]:.4f} - Time: {time.time()-start_time:.2f}s")
            
            if graphing_frequency > 0 and epoch % graphing_frequency == 0:
                validation_times = self.valiloader.dataset[100][0][:, 1].numpy()
                y_of_t_validation = self.valiloader.dataset[100][1][:, 0].numpy()
                f_of_t_validation = self.valiloader.dataset[100][0][:, 0].numpy()
                y_pred_validation = self.model(self.valiloader.dataset[100][0].unsqueeze(0)).detach().numpy()
                
                test_times = self.dataloader.dataset[100][0][:, 1].numpy()
                y_of_t_test = self.dataloader.dataset[100][1][:, 0].numpy()
                f_of_t_test = self.dataloader.dataset[100][0][:, 0].numpy()
                y_pred_test = self.model(self.dataloader.dataset[100][0].unsqueeze(0)).detach().numpy()
                
                valid_mask = ~np.isnan(y_of_t_validation)
                valid_times = validation_times[valid_mask]
                valid_y = y_of_t_validation[valid_mask]
                
                plt.plot(valid_times, valid_y, label='True')
                plt.plot(validation_times, y_pred_validation[0], label='Predicted')
                plt.plot(validation_times, f_of_t_validation, label='Force')
                plt.legend()
                plt.title('Validation')
                plt.show()
                
                valid_mask = ~np.isnan(y_of_t_test)
                valid_times = test_times[valid_mask]
                valid_y = y_of_t_test[valid_mask]
                
                plt.plot(valid_times, valid_y, label='True')
                plt.plot(test_times, y_pred_test[0], label='Predicted')
                plt.plot(test_times, f_of_t_test, label='Force')
                plt.legend()
                plt.title('Training')
                plt.show()
            
    def graph_loss(self):
        plt.plot(self.train_loss, label='train')
        plt.plot(self.vali_loss, label='vali')
        plt.legend()
        plt.show()
        
if __name__ == '__main__':
    LNO_training_dataset = pd.read_csv('Training Data\Duffing Oscillator\duffing_oscillator_training_data.csv')
    #LNO_sparse_training_dataset = fd.SparsePandasDataset(LNO_training_dataset, .01, 0)
    LNO_sparse_training_dataset = fd.PandasDataset(LNO_training_dataset)
    LNO_training_dataloader = torch.utils.data.DataLoader(LNO_sparse_training_dataset, batch_size=32, shuffle=True)
    LNO_validation_dataset = pd.read_csv('Training Data\Duffing Oscillator\duffing_oscillator_test_data.csv')
    #LNO_sparse_validation_dataset = fd.SparsePandasDataset(LNO_validation_dataset, .1, 1)
    LNO_validation_dataset = fd.PandasDataset(LNO_validation_dataset)
    LNO_validation_dataloader = torch.utils.data.DataLoader(LNO_validation_dataset, batch_size=32, shuffle=True)

    LNO_trainer_object = LNO_trainer(4, 16, LNO_training_dataloader, LNO_validation_dataloader)
    LNO_trainer_object.train(1000, 1e-3, graphing_frequency=150)
    LNO_trainer_object.graph_loss()

        
        