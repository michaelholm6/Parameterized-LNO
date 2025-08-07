from matplotlib import pyplot as plt
import numpy as np
import torch
from scipy.integrate import solve_ivp
import pandas as pd
import sys

np.set_printoptions(threshold=sys.maxsize)

def wave_equation(t, y, A, omega):
    y_prime = A*np.cos(omega*t)
    return y_prime
    
def wave_solution(y_zero, t_final, steps, A, omega):
    t = np.linspace(0, t_final, steps)
    solution = solve_ivp(wave_equation, [t[0], t[-1]], y_zero, t_eval=t, args=(A, omega))
    return solution.y[0], solution.t

def genrate_wave_data(y_zero, t_final, steps, A, omega):
    x_of_t, x_t = wave_solution(y_zero, t_final, steps, A, omega)
    f_of_t = A*np.cos(omega*np.linspace(0, t_final, steps))
    f_t = np.linspace(0, t_final, steps)
    return x_of_t, f_of_t, x_t, f_t

def test_duffing_oscillator(A, omega, t_final, steps, y_zero):
    x_of_t, f_of_t, x_t, f_t = genrate_wave_data(y_zero, t_final, steps, A, omega)
    plt.plot(x_t, x_of_t, label='True')
    plt.plot(f_t, f_of_t, label='Force')
    plt.legend()
    plt.show()

def generate_wave_training_data(a_min_traing, a_max_train, a_step_size_traing, a_min_test, a_max_test, a_step_size_test, omega, y_zero, t_final, steps):
    training_dataframe = pd.DataFrame(columns=['f_of_t', 'x_of_t', 'f_t', 'x_t'])
    test_dataframe = pd.DataFrame(columns=['f_of_t', 'x_of_t', 'f_t', 'x_t'])
    a_train = np.arange(a_min_traing, a_max_train, a_step_size_traing)
    a_test = np.arange(a_min_test, a_max_test, a_step_size_test)
    
    for a in a_train:
        x_of_t, f_of_t, x_t, f_t = genrate_wave_data(y_zero, t_final, steps, a, omega)
        training_dataframe.loc[len(training_dataframe)] = [np.array(f_of_t), np.array(x_of_t), np.array(f_t), np.array(x_t)]
        
    for a in a_test:
        x_of_t, f_of_t, x_t, f_t = genrate_wave_data(y_zero, t_final, steps, a, omega)
        test_dataframe.loc[len(test_dataframe)] = [np.array(f_of_t), np.array(x_of_t), np.array(f_t), np.array(x_t)]
        
    training_dataframe.to_csv('Training Data\Wave Data\wave_training_data.csv', float_format='%.10f')
    test_dataframe.to_csv('Training Data\Wave Data\wave_test_data.csv', float_format='%.10f')
    
generate_wave_training_data(1, 10, 1, 1, 10, 1, 5, [0, 0], 10, 2000)

#test_duffing_oscillator(5, 5, 20, 500, [0, 0])