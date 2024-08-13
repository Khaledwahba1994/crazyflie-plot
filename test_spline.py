import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
import cfusdlog

# Assume cfusdlog and data are properly configured and loaded
f = "data/test_data/cf3_t_04"
data = cfusdlog.decode(f)['fixedFrequency']
starttime = data['timestamp'][0]
time = ((data['timestamp'] - starttime)/1000.0).tolist()


for key in data.keys():
    if "acc.x" in key:
        acc = np.array([data["acc.x"] ,data["acc.y"], data["acc.z"]])
    elif "stateEstimate.ax" in key:
        acc_w = np.array([data["stateEstimate.ax"] ,data["stateEstimate.ay"], data["stateEstimate.az"]])

    elif "stateEstimateZ.x" in key:
        pos = np.array([data["stateEstimateZ.x"] ,data["stateEstimateZ.y"], data["stateEstimateZ.z"]])/1000.0

    elif "stateEstimateZ.x" in key:
        vel = np.array([data["stateEstimateZ.vx"] ,data["stateEstimateZ.vy"], data["stateEstimateZ.vz"]])/1000.0

    else:
        continue

data_xyz = []
for i in range(3):
    data_i = np.vstack((time, pos[i])).T
    data_xyz.append(data_i)

# Polynomial degree and number of segments
degree = 5
num_segments = 50

for data_points in data_xyz:
    num_points = len(data_points)
    len_segment = int(np.ceil(num_points / num_segments))
    coeffs = [cp.Variable(degree + 1) for _ in range(num_segments)]
    cost = 0
    constraints = []

    # Create cost function and constraints for each segment
    for i in range(num_segments):
        start_id = i * (len_segment)
        end_id = min(start_id + len_segment, num_points)
        
        # Select segment points
        x_data = data_points[start_id:end_id, 0]
        y_data = data_points[start_id:end_id, 1]
        
        # Vandermonde matrix for polynomial terms
        X = np.vander(x_data, N=degree+1, increasing=True)
        y_est = cp.matmul(X, coeffs[i])
        
        # Objective: Minimize the sum of squared differences
        cost += 100*cp.sum_squares(y_est - y_data) + 0.00001* cp.sum_squares(coeffs[i])
        
        # Add derivative constraints at the boundaries
        if i < num_segments - 1:
            x_transition = data_points[end_id - 1, 0]
            # Value and derivatives at the boundary
            boundary_values = [sum(coeffs[i][d] * x_transition**d for d in range(degree + 1)),
                               sum(d * coeffs[i][d] * x_transition**(d - 1) for d in range(1, degree + 1)),
                               sum(d * (d - 1) * coeffs[i][d] * x_transition**(d - 2) for d in range(2, degree + 1)),
                            #    sum(d * (d - 1) * (d - 2) * coeffs[i][d] * x_transition**(d - 3) for d in range(3, degree + 1)),
                                ]
            boundary_values_next = [sum(coeffs[i + 1][d] * x_transition**d for d in range(degree + 1)),
                                    sum(d * coeffs[i + 1][d] * x_transition**(d - 1) for d in range(1, degree + 1)),
                                    sum(d * (d - 1) * coeffs[i + 1][d] * x_transition**(d - 2) for d in range(2, degree + 1)),
                                    # sum(d * (d - 1) * (d - 2) * coeffs[i + 1][d] * x_transition**(d - 3) for d in range(3, degree + 1)),
                                    ]

            # Constraints for continuity and smoothness
            for val, val_next in zip(boundary_values, boundary_values_next):
                constraints.append(val == val_next)

    # Solve the optimization problem
    problem = cp.Problem(cp.Minimize(cost), constraints)
    problem.solve(solver=cp.OSQP)

    # Plotting
    plt.figure(figsize=(10, 6))
    for i in range(num_segments):
        x_vals = np.linspace(data_points[i * len_segment, 0], data_points[min((i + 1) * len_segment, num_points) - 1, 0], 100)
        X_plot = np.vander(x_vals, N=degree+1, increasing=True)
        y_vals = X_plot.dot(coeffs[i].value)
        plt.plot(x_vals, y_vals, label=f'Segment {i+1}')

    plt.scatter(data_points[:, 0], data_points[:, 1], color='red', label='Data Points', s=2)
    plt.title('Piecewise Polynomial Regression with Smoothness Constraints')
    plt.legend()
    plt.show()


plotthis = True
if plotthis == True and problem.status == cp.OPTIMAL:
    plt.figure(figsize=(15, 10))

    # Calculate derivatives for each segment and plot them
    for i in range(num_segments):
        # Define the range of x values for plotting
        x_vals = np.linspace(data_points[i * len_segment, 0], data_points[min((i + 1) * len_segment, num_points) - 1, 0], 100)
        X_plot = np.vander(x_vals, N=degree+1, increasing=True)

        # Get the coefficients of the current segment
        current_coeffs = coeffs[i].value

        # Compute the polynomial values (already done)
        y_vals = X_plot @ current_coeffs

        # Compute the second derivative
        second_derivative_coeffs = [(d * (d - 1)) * current_coeffs[d] for d in range(2, degree + 1)]
        second_derivative = np.polyval(second_derivative_coeffs[::-1], x_vals)

        # Compute the third derivative
        third_derivative_coeffs = [(d * (d - 1) * (d - 2)) * current_coeffs[d] for d in range(3, degree + 1)]
        third_derivative = np.polyval(third_derivative_coeffs[::-1], x_vals)

        # Plotting the original polynomial
        plt.subplot(3, 1, 1)
        plt.plot(x_vals, y_vals, label=f'Segment {i+1}')
        plt.title('Polynomial Fit')
        plt.legend()

        # Plotting the second derivative
        plt.subplot(3, 1, 2)
        plt.plot(x_vals, second_derivative, label=f'Segment {i+1} Second Derivative')
        plt.title('Second Derivative')
        plt.legend()

        # Plotting the third derivative
        plt.subplot(3, 1, 3)
        plt.plot(x_vals, third_derivative, label=f'Segment {i+1} Third Derivative')
        plt.title('Third Derivative')
        plt.legend()

    plt.tight_layout()
    plt.show()
else:
    # print("Optimization problem did not solve optimally.")
    print(".")