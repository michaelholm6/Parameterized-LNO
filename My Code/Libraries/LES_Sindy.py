from torch import nn
import torch
import math
import numpy as np
import time
from matplotlib import pyplot as plt
import functional_dataloaders as fd
import pandas as pd
from scipy.integrate import trapezoid
from findiff import Diff
import time

class NormalizedMSELoss_and_norm_1():
    def __init__(self, gamma):
        self.gamma = gamma
        
    def forward(self, y_pred, y_true, coefficients):
        # Normalized MSE loss
        mask = ~torch.isnan(y_true)
        y_pred_valid = y_pred[mask]
        y_true_valid = y_true[mask]
        
        # Normalized MSE loss (only using valid (non-NaN) values)
        loss = torch.mean((y_pred_valid - y_true_valid)**2) / (torch.mean(y_true_valid**2) + 1e-10)
        loss += self.gamma*torch.sum(torch.abs(coefficients))
        return loss
    
    def __call__(self, y_pred, y_true, coefficients):
        return self.forward(y_pred, y_true, coefficients)

class LES_SINDy_model():
    def __init__(self, functions, deriv_numbers, s_evaluation_points, *args, **kwargs):  
        self.coefficients = []
        self.functions = functions
        self.deriv_numbers = deriv_numbers
        self.s_evaluation_points = s_evaluation_points
        
    def __call__(self, x):
        output = torch.zeros(len(self.s_evaluation_points))
        for j in range(len(self.s_evaluation_points)):
            for k in range(x.shape[1]):
                output[j] += (self.coefficients[k]*x[j, k]).item()
        return output
    
class LES_SINDy_trainer():
    def __init__(self, model, dataloader, valiloader, gamma):
        self.model = model
        self.dataloader = dataloader
        self.valiloader = valiloader
        self.s_evaluition_points = model.s_evaluation_points
        self.gamma = gamma
        
        self.laplace_space_training_data_lhs = torch.zeros(len(self.dataloader) * self.dataloader.batch_size, len(model.s_evaluation_points), len(model.functions))
        self.laplace_space_training_data_rhs = torch.zeros(len(self.dataloader) * self.dataloader.batch_size, len(model.s_evaluation_points))
        
        for batch, (x, y) in enumerate(self.dataloader):
                print('Batch:', batch)
                h = y[0, 1, 1] - y[0, 0, 1]
                for k, s in enumerate(self.s_evaluition_points):
                    for i in range(x.shape[0]):
                        if not torch.all(x[i, :, 0] == 0):
                            laplace_transform = [x[i, l, 0]*math.exp(-s*(y[i, l, 1] - y[i, 0, 1])) for l in range(y.shape[1])]
                            self.laplace_space_training_data_rhs[batch * self.dataloader.batch_size + i, k] = trapezoid(laplace_transform, dx=h)
                for j, s in enumerate(self.s_evaluition_points):
                    for k, function in enumerate(model.functions):
                        for i in range(x.shape[0]):
                            laplace_transform = [function(y[i, l, 0])*math.exp(-s*(y[i, l, 1] - y[i, 0, 1])) for l in range(y.shape[1])]
                            self.laplace_space_training_data_lhs[batch * self.dataloader.batch_size + i, j, k] = trapezoid(laplace_transform, dx=h) * s**(model.deriv_numbers[k])
                            for l in range(model.deriv_numbers[k]):
                                if l == 0:
                                    self.laplace_space_training_data_lhs[batch * self.dataloader.batch_size + i, j, k] -= s**(model.deriv_numbers[k]-l-1)*function(y[i, 0, 0])
                                else:
                                    differentiator = Diff(0, float(h)) ** l 
                                    self.laplace_space_training_data_lhs[batch * self.dataloader.batch_size + i, j, k] -= s**(model.deriv_numbers[k]-l-1)*differentiator(function(y[i, :, 0]).numpy())[0]
                            
        self.laplace_space_test_data_lhs = torch.zeros(len(self.valiloader) * self.valiloader.batch_size, len(model.s_evaluation_points), len(model.functions))
        self.laplace_space_test_data_rhs = torch.zeros(len(self.valiloader) * self.valiloader.batch_size, len(model.s_evaluation_points))
        
        for batch, (x, y) in enumerate(self.valiloader):
            print('Batch:', batch)
            for k, s in enumerate(self.s_evaluition_points):
                for i in range(x.shape[0]):
                    laplace_transform = [x[i, l, 0]*math.exp(-s*(y[i, l, 1] - y[i, 0, 1])) for l in range(y.shape[1])]
                    self.laplace_space_test_data_rhs[batch * self.dataloader.batch_size + i, k] = trapezoid(laplace_transform, dx=h)
            for j, s in enumerate(self.s_evaluition_points):
                for k, function in enumerate(model.functions):
                    for i in range(x.shape[0]):
                        laplace_transform = [function(y[i, l, 0])*math.exp(-s*(y[i, l, 1] - y[i, 0, 1])) for l in range(y.shape[1])]
                        self.laplace_space_test_data_lhs[batch * self.dataloader.batch_size + i, j, k] = trapezoid(laplace_transform, dx=h) * s**(model.deriv_numbers[k])
                        for l in range(model.deriv_numbers[k]):
                            if l == 0:
                                self.laplace_space_test_data_lhs[batch * self.dataloader.batch_size + i, j, k] -= s**(model.deriv_numbers[k]-l-1)*function(y[i, 0, 0])
                            else:
                                differentiator = Diff(0, float(h))**l
                                self.laplace_space_test_data_lhs[batch * self.dataloader.batch_size + i, j, k] -= s**(model.deriv_numbers[k]-l-1)*differentiator(function(y[i, :, 0]).numpy())[0]
        
        deleting_zeros = True
        while deleting_zeros:
            deleting_zeros = False
            for i, row in enumerate(self.laplace_space_training_data_rhs):
                if torch.all(row == 0):
                    self.laplace_space_training_data_rhs = torch.cat((self.laplace_space_training_data_rhs[:i, :], self.laplace_space_training_data_rhs[i+1:, :]))
                    self.laplace_space_training_data_lhs = torch.cat((self.laplace_space_training_data_lhs[:i, :], self.laplace_space_training_data_lhs[i+1:, :]))
                    deleting_zeros = True
                    break
            
                            
    def calculate_initial_derivative(self, function_values, degree, h):
        if degree == 0:
            return function_values[0]
        if degree == 1:
            answer = (function_values[1] - function_values[0]) / h
        answer = 1/(h**degree)*torch.sum(torch.tensor([(-1)**i*math.comb(degree, i)*function_values[i] for i in range(degree+1)]))
        return answer
        
    def train(self):
        with torch.no_grad():
            start_time = time.time()
            
            self.model.coefficients = torch.linalg.lstsq(self.laplace_space_training_data_lhs, self.laplace_space_training_data_rhs).solution
            self.model.coefficients = torch.mean(self.model.coefficients, axis=0)
            
            train_loss = 0
            vali_loss = 0
            
            loss_fn = NormalizedMSELoss_and_norm_1(self.gamma)
            
            for i, datapoint in enumerate(self.laplace_space_training_data_lhs):
                y = datapoint
                rhs_laplace = self.laplace_space_training_data_rhs[i]
                rhs_laplace_predicted = self.model(y)
                loss = loss_fn(rhs_laplace_predicted, rhs_laplace, self.model.coefficients)
                loss += loss.item()
            
                train_loss += loss.item()
                
            for i, datapoint in enumerate(self.laplace_space_test_data_lhs):
                y = self.laplace_space_test_data_lhs[i]
                rhs_laplace = self.laplace_space_test_data_rhs[i]
                rhs_laplace_predicted = self.model(y)
                loss = loss_fn(rhs_laplace_predicted, rhs_laplace, self.model.coefficients)
                
                vali_loss += loss.item()
                
            self.train_loss = train_loss/len(self.laplace_space_training_data_lhs)
            self.vali_loss = vali_loss/len(self.laplace_space_test_data_lhs)
        
            print(f'Coefficients: {self.model.coefficients}, Training Time: {time.time() - start_time}, Training Loss: {self.train_loss}, Validation Loss: {self.vali_loss}')


if __name__ == "__main__":
    
    def x(x):
        return x
    
    def x_cubed(x):
        return torch.pow(x, 3)
    
    gamma = 0
    LES_SINDy_training_dataset = pd.read_csv('Training Data\Duffing Oscillator\duffing_oscillator_training_data.csv')
    LES_SINDy_sparse_training_dataset = fd.PandasDataset(LES_SINDy_training_dataset)
    LES_SINDy_training_dataloader = torch.utils.data.DataLoader(LES_SINDy_sparse_training_dataset, batch_size=8, shuffle=False)
    LES_SINDy_validation_dataset = pd.read_csv('Training Data\Duffing Oscillator\duffing_oscillator_test_data.csv')
    LES_SINDy_validation_dataset = fd.PandasDataset(LES_SINDy_validation_dataset)
    LES_SINDy_validation_dataloader = torch.utils.data.DataLoader(LES_SINDy_validation_dataset, batch_size=8, shuffle=False)
    
    LES_SINDy_model_instance = LES_SINDy_model([x, x, x, x_cubed], [0, 1, 2, 0], np.arange(1, 5, .3))
    #LES_SINDy_model_instance.coefficients = nn.Parameter(torch.tensor([1, .3, 1]))
    LES_SINDy_trainer_object = LES_SINDy_trainer(LES_SINDy_model_instance, LES_SINDy_training_dataloader, LES_SINDy_validation_dataloader, gamma)
    #LES_SINDy_trainer_object = LES_SINDy_trainer(LES_SINDy_model([x, x, x, x_cubed], [0, 1, 2, 0], np.arange(.1, 1.5, .1)), LES_SINDy_training_dataloader, LES_SINDy_validation_dataloader, gamma)
    LES_SINDy_trainer_object.train()
            