import numpy as np
import matplotlib.pyplot as plt

def system_of_equations(t, y, a, b, c, d):
    dy1dt = a*y[0] - b*y[0]*y[1]
    dy2dt = c*y[0]*y[1] - d*y[1]
    return np.array([dy1dt, dy2dt])

def runge_kutta_2nd_order(y, t, h, a,b,c,d):
    k1 = h * system_of_equations(t, y, a,b,c,d)
    k2 = h * system_of_equations(t + h, y + k1,a,b,c,d)
    y_next = y + 0.5 * (k1 + k2)
    return y_next

def euler_method_step(y, t, h, a, b, c, d):
    """
    Funzione di un passo del metodo di Eulero.
    """
    y_next = y + h * system_of_equations(t, y, a, b, c, d)
    return y_next

def euler_method(y0, t, h, a, b, c, d):
    """
    Metodo di Eulero per risolvere un sistema di equazioni differenziali ordinarie (ODE).
    """
    num_steps = len(t)
    y_values = np.zeros((num_steps, len(y0)))
    y_values[0] = y0

    for i in range(1, num_steps):
        y_values[i] = euler_method_step(y_values[i-1], t[i-1], h, a, b, c, d)

    return y_values


def generate_dataset(T, a_values, b_values, c_values, d_values, h, fidelity):
    num_steps = int(T / h)
    time_points = np.linspace(0, T, num_steps + 1)
    y_values_list = []

    if fidelity=="HF":
        for j in range(len(a_values)):
            y_values = np.zeros((num_steps + 1, 2))
            y_values[0, :] = 0.5

            for i in range(num_steps):
                y_values[i + 1, :] = runge_kutta_2nd_order(y_values[i, :], time_points[i], h, a_values[j],b_values[j],c_values[j],d_values[j])

            y_values_list.append(y_values)
    else:
        for j in range(len(a_values)):
            y_values = np.zeros((num_steps + 1, 2))
            y_values[0, :] = 0.5

            for i in range(num_steps):
                y_values[i + 1, :] = euler_method_step(y_values[i, :], time_points[i], h, a_values[j],b_values[j],c_values[j],d_values[j])

            y_values_list.append(y_values)

    return time_points, y_values_list, a_values, b_values, c_values, d_values

def save_dataset(time_points, y_values_list, a_values,b_values,c_values,d_values, filename='dataset_abcd_variationLLF.csv'):
    data = []

    for i in range(len(a_values)):
        for j, t in enumerate(time_points):
            row = [t, y_values_list[i][j, 0], y_values_list[i][j, 1], a_values[i],b_values[i],c_values[i],d_values[i]]
            data.append(row)

    data = np.array(data)
    np.savetxt(filename, data, delimiter=',')

# Parameters
T = 6.0
# 0.25,0.15,0.01 <- h per LLF, LF e HF
h = 0.02

# Array separati per ciascun parametro
# a_values = np.array([0.1, 0.2, 0.15, 0.12, 0.08])
# b_values = np.array([0.02, 0.03, 0.025, 0.018, 0.015])
# c_values = np.array([0.01, 0.015, 0.012, 0.008, 0.007])
# d_values = np.array([0.1, 0.2, 0.15, 0.12, 0.08])

# a_values = np.array([2.5,   2.5,    2.5,    2.,      2.5,    2.5,      2])
# b_values = np.array([1.,    1.4,    1.,     0.6,    1.4,    0.7,     0.7])
# c_values = np.array([1.,    0.6,    1.4,    1.4,    1.4,    1.,      1.])
# d_values = np.array([1.2,   1.5,    1.2,    1.5,    1.2,    1.5,    1.2 ])

a_values=np.array([2., 2.3,  2.6 ,  2.9, 3.2 ])
b_values = np.array([0.6, 0.9, 1.2,1.5,1.9])
c_values = np.array([0.6, 0.9, 1.2,1.5,1.9])
d_values = np.array([0.8, 1.1, 1.4,1.7,2.1])


# Generate and save dataset for different values of mu
time_points, y_values_list, a_values, b_values, c_values, d_values = generate_dataset(T, a_values,b_values,c_values,d_values, h,"LF")
save_dataset(time_points, y_values_list, a_values,b_values,c_values,d_values,)

# Plot results for each value of mu (optional)
plt.figure(figsize=(12, 8))

for i in range(len(a_values)):
    plt.plot(time_points, y_values_list[i][:, 0], label=f'y1(t) for a={a_values[i]}, b={b_values[i]}, c={c_values[i]}, d={d_values[i]}',marker='o')

plt.xlabel('Time')
plt.ylabel('Values')
plt.legend()
plt.title('Second Order Runge-Kutta Method for the System of Differential Equations for Different Values of parameters')
plt.show()


h2 = 0.01

# Generate and save dataset for different values of mu
time_points2, y_values_list2, a_values2, b_values2, c_values2, d_values2 = generate_dataset(T, a_values,b_values,c_values,d_values, h2,"HF")
save_dataset(time_points2, y_values_list2, a_values2,b_values2,c_values2,d_values2,"dataset_abcd_variationHF.csv")

# Plot results for each value of mu (optional)
plt.figure(figsize=(12, 8))

for i in range(len(a_values)):
    plt.plot(time_points2, y_values_list2[i][:, 0], label=f'y1(t) for a={a_values2[i]}, b={b_values2[i]}, c={c_values2[i]}, d={d_values2[i]}',marker='o')

plt.xlabel('Time')
plt.ylabel('Values')
plt.legend()
plt.title('Second Order Runge-Kutta Method for the System of Differential Equations for Different Values of parameters')
plt.show()

print(np.sum((np.array(y_values_list2)- np.array(y_values_list))**2))