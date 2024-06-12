import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt
import cfusdlog


# f = "data/test_data/cf3_t_05" # add the path of your data
f = "/home/khaledwahba94/imrc/crazyflie-plot/data/2cfs_forest/cf2/opt/cf2_1_01"
data  = cfusdlog.decode(f)['fixedFrequency'] 
starttime = data['timestamp'][0] 
time = ((data['timestamp'] - starttime)/1000.0).tolist()

for key in data.keys():
    if "acc.x" in key:
        acc = np.array([data["acc.x"] ,data["acc.y"], data["acc.z"]])
    elif "stateEstimateZ.px" in key:
        pos = np.array([data["stateEstimateZ.px"] ,data["stateEstimateZ.py"], data["stateEstimateZ.pz"]])/1000.0
    elif "stateEstimateZ.vx" in key:
        vel = np.array([data["stateEstimateZ.pvx"] ,data["stateEstimateZ.pvy"], data["stateEstimateZ.pvz"]])/1000.0
    else:
        continue
data_xyz = []
for i in range(3):
    data_i = np.zeros((4,len(time)))
    data_i[0,:] = time
    data_i[1,:] = pos[i,:]
    data_i[2,:] = vel[i,:]
    data_xyz.append(data_i.T)

degree = 4
num_segments = 50
a = 400 # weight for the least square error 
l = 0.00001 # weight for the regularization

for data_points in data_xyz:
    coeffs = [cp.Variable(degree + 1) for _ in range(num_segments)]
    num_points = len(data_points)
    len_segment = int(np.ceil(num_points/num_segments))
    cost = 0
    x_vals = []
    constraints = []

    for i in range(num_segments):
        start_id = i*(len_segment-1)
        end_id   =  min(start_id + len_segment, num_points)
        # if end_id == num_points-1:
        #     end_id+=1
        for j in range(start_id, end_id):
            point = data_points[j]
            x = point[0]
            y = point[1]
            y_poly = sum([coeffs[i][d]*x**d for d in range(degree+1)])
            cost += a*cp.sum_squares(y_poly - y)
        cost += l * cp.sum_squares(coeffs[i])
        x_vals.append(data_points[start_id:end_id - 1,0])

        if i < num_segments - 1:
            x = data_points[end_id - 1][0]
            # Calculate the derivatives at the boundary
            boundary_value    = sum(coeffs[i][d]*x**d for d in range(degree + 1))
            dboundary_value   = sum(d * coeffs[i][d] * x**(d - 1) for d in range(1, degree + 1))
            ddboundary_value  = sum(d * (d - 1) * coeffs[i][d] * x**(d - 2) for d in range(2, degree + 1))
            boundary_value_next   = sum(coeffs[i + 1][d] * x**d for d in range(degree + 1))
            dboundary_value_next  = sum(d * coeffs[i + 1][d] * x**(d - 1) for d in range(1, degree + 1))
            ddboundary_value_next = sum(d * (d - 1) * coeffs[i + 1][d] * x**(d - 2) for d in range(2, degree + 1))
           # add constraints
            constraints.append(boundary_value == boundary_value_next)
            constraints.append(dboundary_value == dboundary_value_next)
            constraints.append(ddboundary_value == ddboundary_value_next)
        
            if degree > 3:
                dddboundary_value  = sum(d * (d - 1) * (d - 2) * coeffs[i][d] * x**(d - 3) for d in range(3, degree + 1))
                dddboundary_value_next = sum(d * (d - 1) * (d - 2) * coeffs[i + 1][d] * x**(d - 3) for d in range(3, degree + 1))
                constraints.append(dddboundary_value == dddboundary_value_next)

            
    problem = cp.Problem(cp.Minimize(cost), constraints)
    problem.solve(solver=cp.OSQP)

    # Plotting
    fig, (ax1, ax2, ax3) = plt.subplots(3,1,figsize=(10, 18), sharex=True, sharey=True)
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
        ax1.plot(x_vals[i], y_vals[i], label=f'Segment {i+1}', linewidth=2)
        ax2.plot(x_vals[i], y_der_vals[i], label=f'Segment {i+1}', linewidth=2)
        ax3.plot(x_vals[i], y_dder_vals[i], label=f'Segment {i+1}', linewidth=2)

    # Plot data items
    ax1.plot(data_points[:, 0], data_points[:, 1], color='blue', label='Data', linewidth=0.5)
    ax2.plot(data_points[:, 0], data_points[:, 2], color='blue', label='Data', linewidth=0.5)
    ax1.grid()
    ax2.grid()
    ax3.grid()
    # ax3.plot(data_points[:, 0], data_points[:, 3], color='blue', label='Data', linewidth=1)
    ax1.set_title('Piecewise Polynomial Regression')
    # ax1.legend()
    # ax2.legend()
    plt.show()

    