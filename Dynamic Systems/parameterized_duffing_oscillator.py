from matplotlib import pyplot as plt
import numpy as np
import torch
from scipy.integrate import solve_ivp
import pandas as pd
import sys
import os

np.set_printoptions(threshold=sys.maxsize)

def duffing_oscillator_governing_equation(t, y, c, A, decay, omega):
    y_one_prime = y[1]
    y_two_prime = -y[0] - c*y[1] + A*np.exp(-decay*t)*np.sin(omega*t) - y[0]**3
    return [y_one_prime, y_two_prime]
    
def duffing_oscillator_solution(y_zero, t_final, steps, c, A, decay, omega):
    t = np.linspace(0, t_final, steps)
    solution = solve_ivp(duffing_oscillator_governing_equation, [t[0], t[-1]], y_zero, t_eval=t, args=(c, A, decay, omega))
    return solution.y[0], solution.t

def genrate_duffing_oscillator_data(y_zero, t_final, steps, c, A, decay, omega):
    x_of_t, x_t = duffing_oscillator_solution(y_zero, t_final, steps, c, A, decay, omega)
    f_of_t = A*np.exp(-decay*np.linspace(0, t_final, steps))*np.sin(omega*np.linspace(0, t_final, steps))
    f_t = np.linspace(0, t_final, steps)
    return x_of_t, f_of_t, x_t, f_t

def test_duffing_oscillator(A, omega, t_final, steps, c, decay, y_zero):
    x_of_t, f_of_t, x_t, f_t = genrate_duffing_oscillator_data(y_zero, t_final, steps, c, A, decay, omega)
    x_first_derivs = np.gradient(x_of_t, x_t)
    x_second_derivs = np.gradient(x_first_derivs, x_t)
    theoretical_zero_line = x_of_t + c*x_first_derivs - A*np.exp(-decay*x_t)*np.sin(omega*x_t) + x_of_t**3 + x_second_derivs
    plt.plot(x_t, x_of_t, label='True')
    plt.plot(f_t, f_of_t, label='Force')
    plt.plot(x_t, theoretical_zero_line, label='Theoretical Zero Line')
    plt.legend()
    plt.show()

def generate_duffing_oscillator_training_data(a_min_traing, a_max_train, a_step_size_traing, a_min_test, a_max_test, a_step_size_test, omega, c, decay, y_zero, t_final, steps):
    training_dataframe = pd.DataFrame(columns=['f_of_t', 'x_of_t', 'f_t', 'x_t', 'parameters'])
    test_dataframe = pd.DataFrame(columns=['f_of_t', 'x_of_t', 'f_t', 'x_t', 'parameters'])
    a_train = np.arange(a_min_traing, a_max_train, a_step_size_traing)
    a_test = np.arange(a_min_test, a_max_test, a_step_size_test)
    
    for a in a_train:
        x_of_t, f_of_t, x_t, f_t = genrate_duffing_oscillator_data(y_zero, t_final, steps, c, a, decay, omega)
        training_dataframe.loc[len(training_dataframe)] = [np.array(f_of_t), np.array(x_of_t), np.array(f_t), np.array(x_t), np.array(c)]
        
    for a in a_test:
        x_of_t, f_of_t, x_t, f_t = genrate_duffing_oscillator_data(y_zero, t_final, steps, c, a, decay, omega)
        test_dataframe.loc[len(test_dataframe)] = [np.array(f_of_t), np.array(x_of_t), np.array(f_t), np.array(x_t), np.array(c)]
        
    # Define folder path
    folder_path = os.path.join("Training Data", "Duffing Oscillator")

    # Ensure the folder exists
    os.makedirs(folder_path, exist_ok=True)

    # File paths
    train_file = os.path.join(folder_path, "parameterized_duffing_oscillator_training_data.csv")
    test_file = os.path.join(folder_path, "parameterized_duffing_oscillator_test_data.csv")

    # Save training data
    if os.path.isfile(train_file):
        training_dataframe.to_csv(train_file, float_format="%.10f", mode="a", header=False)
    else:
        training_dataframe.to_csv(train_file, float_format="%.10f")

    # Save test data
    if os.path.isfile(test_file):
        test_dataframe.to_csv(test_file, float_format="%.10f", mode="a", header=False)
    else:
        test_dataframe.to_csv(test_file, float_format="%.10f")
    
generate_duffing_oscillator_training_data(0.05, 10, .5, .14, 9.09, .1, 5, 0.3, 0.05, [0, 0], 10, 4000)
generate_duffing_oscillator_training_data(0.05, 10, .5, .14, 9.09, .1, 5, 0.1, 0.05, [0, 0], 10, 4000)
generate_duffing_oscillator_training_data(0.05, 10, .5, .14, 9.09, .1, 5, 0.5, 0.05, [0, 0], 10, 4000)
generate_duffing_oscillator_training_data(0.05, 10, .5, .14, 9.09, .1, 5, 0.7, 0.05, [0, 0], 10, 4000)
generate_duffing_oscillator_training_data(0.05, 10, .5, .14, 9.09, .1, 5, 0.9, 0.05, [0, 0], 10, 4000)

