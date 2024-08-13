import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
import cfusdlog

# data_points = np.array([
#     [0, 1], [1, 3], [2, 2], [3, 5], [4, 4], [5, 5], [6, 7], [7, 6], [8, 8], [9, 9]
# ])

# data_points = np.array([
#     [1,1], [2,3], [5,5] ,[10,10], [12,12]
# ])

f = "data/test_data/cf3_t_03"
data  = cfusdlog.decode(f)['fixedFrequency'] 
starttime = data['timestamp'][0] 
time = ((data['timestamp'] - starttime)/1000.0).tolist()

# print(data.keys())
for key in data.keys():
    if "acc.x" in key:
        acc = np.array([data["acc.x"] ,data["acc.y"], data["acc.z"]])
    elif "stateEstimate.ax" in key:
        # print("yas")
        acc_w = np.array([data["stateEstimate.ax"] ,data["stateEstimate.ay"], data["stateEstimate.az"]])
        # print(acc_w)
    elif "stateEstimateZ.x" in key:
        pos = np.array([data["stateEstimateZ.x"] ,data["stateEstimateZ.y"], data["stateEstimateZ.z"]])/1000.0

    elif "stateEstimateZ.vx" in key:
        vel = np.array([data["stateEstimateZ.vx"] ,data["stateEstimateZ.vy"], data["stateEstimateZ.vz"]])/1000.0
        # print(vel)
        # exit()
    else:
        continue
# exit()
data_xyz = []
for i in range(3):
    data_i = np.zeros((4,len(time)))
    data_i[0,:] = time
    data_i[1,:] = pos[i,:]
    data_i[2,:] = vel[i,:]
    data_i[3,:] = acc_w[i,:]
    data_xyz.append(data_i.T)

degree = 3
num_segments = 20

for data_points in data_xyz:
    coeffs = [cp.Variable(degree + 1) for _ in range(num_segments)]
    num_points = len(data_points)
    len_segment = int(np.ceil(num_points/num_segments))
    cost = 0
    x_vals = []
    constraints = []

    for i in range(num_segments):
        start_id = i*(len_segment)
        end_id   =  min(start_id + len_segment, num_points)
        for j in range(start_id, end_id):
            point = data_points[j]
            Ax = sum([coeffs[i][d]*point[0]**d for d in range(degree+1)])
            cost += 150*cp.sum_squares(Ax - point[1])
        cost += 0.001* cp.sum_squares(coeffs[i])
        # x_vals.append(data_points[start_id:end_id - 1,0])
        x_vals.append(np.linspace(data_points[start_id, 0], data_points[min((i + 1) * len_segment, num_points) - 1, 0], 100))
        if i < num_segments - 1:
            x = data_points[end_id - 1][0]
            # Calculate the derivatives at the boundary
            boundary_value    = sum(coeffs[i][d]*x**d for d in range(degree + 1))
            dboundary_value   = sum(d * coeffs[i][d] * x**(d - 1) for d in range(1, degree + 1))
            ddboundary_value  = sum(d * (d - 1) * coeffs[i][d] * x**(d - 2) for d in range(2, degree + 1))
            
            boundary_value_next   = sum(coeffs[i + 1][d] * x**d for d in range(degree + 1))
            dboundary_value_next  = sum(d * coeffs[i + 1][d] * x**(d - 1) for d in range(1, degree + 1))
            ddboundary_value_next = sum(d * (d - 1) * coeffs[i + 1][d] * x**(d - 2) for d in range(2, degree + 1))
           
            constraints.append(boundary_value == boundary_value_next)
            constraints.append(dboundary_value == dboundary_value_next)
            constraints.append(ddboundary_value == ddboundary_value_next)


    problem = cp.Problem(cp.Minimize(cost), constraints)
    problem.solve(solver=cp.OSQP)

    # Plotting
    # fig, (ax1, ax2, ax3) = plt.subplots(3,1,figsize=(10, 18))
    fig, ax1  = plt.subplots(1,1,figsize=(10, 6))
    # fig2, ax2 = plt.figure(figsize=(10, 6))
    y_vals = []
    y_der_vals = []
    y_dder_vals = []
    for coeff, x_val in zip(coeffs,x_vals):
        y_est = [sum([coeff[d].value * x**d  for d in range(degree+1)])  for x in x_val]
        y_est_der = [sum([(d)*coeff[d].value * x**(d-1)  for d in range(1,degree+1)])  for x in x_val]
        y_est_dder = [sum([d*(d-1)*coeff[d].value * x**(d-2)  for d in range(2,degree+1)])  for x in x_val]
        y_vals.append(y_est)
        y_der_vals.append(y_est_der)
        y_dder_vals.append(y_est_dder)


    for i in range(num_segments):
        ax1.plot(x_vals[i], y_vals[i], label=f'Segment {i+1}')
        # ax2.plot(x_vals[i], y_der_vals[i], label=f'Segment {i+1}')
        # ax3.plot(x_vals[i], y_dder_vals[i], label=f'Segment {i+1}')
    # for i in range(num_segments):

    # Plot data items
    ax1.scatter(data_points[:, 0], data_points[:, 1], color='blue', label='Data', s=5)
    # ax2.scatter(data_points[:, 0], data_points[:, 2], color='blue', label='Data', s=5)
    # ax3.scatter(data_points[:, 0], data_points[:, 3], color='blue', label='Data', s=5)
    # ax1.set_title('Piecewise Polynomial Regression')
    # ax1.legend()
    # ax2.legend()
    plt.show()

    





#####################################################################################################
import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
import cfusdlog

# data_points = np.array([
#     [0, 1], [1, 3], [2, 2], [3, 5], [4, 4], [5, 5], [6, 7], [7, 6], [8, 8], [9, 9]
# ])

# data_points = np.array([
#     [1,1], [2,3], [5,5] ,[10,10], [12,12]
# ])

f = "data/test_data/cf3_t_03"
data  = cfusdlog.decode(f)['fixedFrequency'] 
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
    data_i = np.zeros((2,len(time)))
    data_i[0,:] = time
    data_i[1,:] = pos[i,:]
    data_xyz.append(data_i.T)

degree = 5
num_segments = 20

for data_points in data_xyz:
    coeffs = [cp.Variable(degree + 1) for _ in range(num_segments)]
    num_points = len(data_points)
    len_segment = int(np.ceil(num_points/num_segments))
    cost = 0
    x_vals = []
    constraints = []

    for i in range(num_segments):
        start_id = i*(len_segment)
        end_id   =  min(start_id + len_segment, num_points)
        
        for j in range(start_id, end_id):
            point = data_points[j]
            Ax = sum([coeffs[i][d]*point[0]**d for d in range(degree+1)])
            cost += cp.sum_squares(Ax - point[1]) 
        cost += 0.00001* cp.sum_squares(coeffs[i]) 
        x_vals.append(data_points[start_id:end_id - 1,0])
        # x_vals.append(np.linspace(data_points[start_id, 0], data_points[min((i + 1) * len_segment, num_points) - 1, 0], 100))
        if i < num_segments - 1:
            x = data_points[end_id - 1][0] 
            # Calculate the derivatives at the boundary
            Ax_end_i    = sum(coeffs[i][d]*x**d for d in range(degree + 1))
            dAx_end_i   = sum(d * coeffs[i][d] * x**(d - 1) for d in range(1, degree + 1))
            ddAx_end_i  = sum(d * (d - 1) * coeffs[i][d] * x**(d - 2) for d in range(2, degree + 1))
            
            Ax_start_ip1   = sum(coeffs[i + 1][d] * x**d for d in range(degree + 1))
            dAx_start_ip1  = sum(d * coeffs[i + 1][d] * x**(d - 1) for d in range(1, degree + 1))
            ddAx_start_ip1 = sum(d * (d - 1) * coeffs[i + 1][d] * x**(d - 2) for d in range(2, degree + 1))
            constraints.append(Ax_end_i == Ax_start_ip1)
            constraints.append(dAx_end_i == dAx_start_ip1)
            constraints.append(ddAx_end_i == ddAx_start_ip1)


    problem = cp.Problem(cp.Minimize(cost), constraints=constraints)
    problem.solve(solver=cp.OSQP)

    # Plotting
    plt.figure(figsize=(10, 6))
    y_vals = []
    for coeff, x_val in zip(coeffs,x_vals):
        y_est = [sum([coeff[d].value * x**d  for d in range(degree+1)])  for x in x_val]
        # y_est_der = [sum([(d)*coeff[d].value * x**(d-1)  for d in range(1,degree+1)])  for x in x_val]
        # y_est_dder = [sum([d*(d-1)*coeff[d].value * x**(d-2)  for d in range(2,degree+1)])  for x in x_val]
        y_vals.append(y_est)


    for i in range(num_segments):
        plt.plot(x_vals[i], y_vals[i], label=f'Segment {i+1}')

    # Plot data items
    plt.scatter(data_points[:, 0], data_points[:, 1], color='blue', label='Data', s=5)
    plt.title('Piecewise Polynomial Regression')
    plt.legend()
    plt.show()