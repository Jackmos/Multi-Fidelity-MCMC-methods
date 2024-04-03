import numpy as np
import matplotlib.pyplot as plt

def system_of_equations(t, y, mu):
    dy1dt = y[0] * (mu - 0.1*y[0] - 0.5*y[1] - 0.5*y[2])
    dy2dt = y[1] * (-mu + 0.5*y[0] - 0.3*y[2])
    dy3dt = y[2] * (-mu + 0.2*y[0] + 0.5*y[1])
    return np.array([dy1dt, dy2dt, dy3dt])

def runge_kutta_2nd_order(y, t, h, mu):
    k1 = h * system_of_equations(t, y, mu)
    k2 = h * system_of_equations(t + h, y + k1, mu)
    y_next = y + 0.5 * (k1 + k2)
    return y_next

def euler_method_step(y, t, h, mu):
    """
    Funzione di un passo del metodo di Eulero.
    """
    y_next = y + h * system_of_equations(t, y, mu)
    return y_next

def generate_dataset(T, mu_values, h, fidelity):
    num_steps = int(T / h)
    time_points = np.linspace(0, T, num_steps + 1)
    y_values_list = []

    if fidelity=="HF":
        for j in range(len(mu_values)):
            y_values = np.zeros((num_steps + 1, 3))
            y_values[0, :] = 0.5

            for i in range(num_steps):
                y_values[i + 1, :] = runge_kutta_2nd_order(y_values[i, :], time_points[i], h, mu_values[j])

            y_values_list.append(y_values)
    else:
        for j in range(len(mu_values)):
            y_values = np.zeros((num_steps + 1, 3))
            y_values[0, :] = 0.5

            for i in range(num_steps):
                y_values[i + 1, :] = euler_method_step(y_values[i, :], time_points[i], h, mu_values[j])

            y_values_list.append(y_values)

    return time_points, y_values_list, mu_values

def save_dataset(time_points, y_values_list, mu_values, filename='dataset_mu_LF.csv'):
    data = []

    for i, mu in enumerate(mu_values):
        for j, t in enumerate(time_points):
            row = [t, y_values_list[i][j, 0], y_values_list[i][j, 1], y_values_list[i][j, 2], mu]
            data.append(row)

    data = np.array(data)
    np.savetxt(filename, data, delimiter=',')

# Parameters
T = 10.0
h = 0.02
mu_values = np.linspace(1, 5, 10)  # Change the number of values as needed

# Generate and save dataset for different values of mu
time_points, y_values_list, mu_values = generate_dataset(T, mu_values, h,"HF")
save_dataset(time_points, y_values_list, mu_values, 'dataset_mu_HF.csv')

# Plot results for each value of mu (optional)
plt.figure(figsize=(12, 8))

for i, mu in enumerate(mu_values):
    plt.plot(time_points, y_values_list[i][:, 0], label=f'y1(t) for mu={mu}',marker='o')

plt.xlabel('Time')
plt.ylabel('Values')
plt.legend()
plt.title('Second Order Runge-Kutta Method for the System of Differential Equations for Different Values of mu')
plt.show()

# Plot results for each value of mu (optional)
plt.figure(figsize=(12, 8))

for i, mu in enumerate(mu_values):
    plt.plot(time_points, y_values_list[i][:, 1], label=f'y2(t) for mu={mu}',marker='o')

plt.xlabel('Time')
plt.ylabel('Values')
plt.legend()
plt.title('Second Order Runge-Kutta Method for the System of Differential Equations for Different Values of mu')
plt.show()


# Plot results for each value of mu (optional)
plt.figure(figsize=(12, 8))

for i, mu in enumerate(mu_values):
    plt.plot(time_points, y_values_list[i][:, 2], label=f'y3(t) for mu={mu}',marker='o')

plt.xlabel('Time')
plt.ylabel('Values')
plt.legend()
plt.title('Second Order Runge-Kutta Method for the System of Differential Equations for Different Values of mu')
plt.show()

# -------------------------------------------------
# Parameters
h2 = 0.05

# Generate and save dataset for different values of mu
time_points2, y_values_list2, mu_values2 = generate_dataset(T, mu_values, h2,"LF")
save_dataset(time_points2, y_values_list2, mu_values2, 'dataset_mu_LF.csv')

# Plot results for each value of mu (optional)
plt.figure(figsize=(12, 8))

for i, mu in enumerate(mu_values2):
    plt.plot(time_points2, y_values_list2[i][:, 0], label=f'y1(t) for mu={mu}',marker='o')

plt.xlabel('Time')
plt.ylabel('Values')
plt.legend()
plt.title('Second Order Runge-Kutta Method for the System of Differential Equations for Different Values of mu')
plt.show()

# Plot results for each value of mu (optional)
plt.figure(figsize=(12, 8))

for i, mu in enumerate(mu_values2):
    plt.plot(time_points2, y_values_list2[i][:, 1], label=f'y2(t) for mu={mu}',marker='o')

plt.xlabel('Time')
plt.ylabel('Values')
plt.legend()
plt.title('Second Order Runge-Kutta Method for the System of Differential Equations for Different Values of mu')
plt.show()


# Plot results for each value of mu (optional)
plt.figure(figsize=(12, 8))

for i, mu in enumerate(mu_values2):
    plt.plot(time_points2, y_values_list2[i][:, 2], label=f'y3(t) for mu={mu}',marker='o')

plt.xlabel('Time')
plt.ylabel('Values')
plt.legend()
plt.title('Second Order Runge-Kutta Method for the System of Differential Equations for Different Values of mu')
plt.show()