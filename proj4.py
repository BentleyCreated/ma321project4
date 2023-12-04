import numpy as np
import matplotlib.pyplot as plt

# Define the exact solutions for both IVPs
def exact_solution_ivp1(t):
    return 1 + 0.5 * np.exp(-4*t) - 0.5 * np.exp(-2*t)

def exact_solution_ivp2(t):
    return np.exp(t/2) * np.sin(5*t)

# Define the derivatives for both IVPs
def derivative_ivp1(t, x):
    return 2 - 2*x - np.exp(-4*t)

def derivative_ivp2(t, x):
    return x + 5 * np.exp(t/2) * np.cos(5*t) - 0.5 * np.exp(t/2) * np.sin(5*t)

# Euler's method function
def euler_method(derivative, initial_x, step_size, t_range):
    t_values = np.arange(t_range[0], t_range[1] + step_size, step_size)
    x_values = np.zeros(t_values.shape)
    x_values[0] = initial_x
    
    for i in range(1, len(t_values)):
        t = t_values[i-1]
        x = x_values[i-1]
        x_values[i] = x + step_size * derivative(t, x)
    
    return t_values, x_values

# Set the range for t
t_range = (0, 5)

# Step sizes to use
step_sizes = [0.1, 0.05, 0.01]

# Plotting the results for IVP1 and IVP2 with different step sizes
fig, axs = plt.subplots(len(step_sizes), 2, figsize=(10, 10))
fig.suptitle('Figure 1: Initial value problem')

for i, h in enumerate(step_sizes):
    # IVP1
    t_vals, x_vals_euler_ivp1 = euler_method(derivative_ivp1, 1, h, t_range)
    axs[i, 0].plot(t_vals, x_vals_euler_ivp1, 'o-', label='Euler', markersize=4)
    axs[i, 0].plot(t_vals, exact_solution_ivp1(t_vals), 'r', label='Exact')
    axs[i, 0].set_title(f'IVP1 with h = {h}')
    axs[i, 0].legend()
    
    # IVP2
    t_vals, x_vals_euler_ivp2 = euler_method(derivative_ivp2, 0, h, t_range)
    axs[i, 1].plot(t_vals, x_vals_euler_ivp2, 'o-', label='Euler', markersize=4)
    axs[i, 1].plot(t_vals, exact_solution_ivp2(t_vals), 'r', label='Exact')
    axs[i, 1].set_title(f'IVP2 with h = {h}')
    axs[i, 1].legend()

# Adjusting layout
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
