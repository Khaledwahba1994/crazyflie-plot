import cfusdlog
import yaml
import numpy as np
import subprocess
import rowan as rn
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import SubplotSpec, GridSpec
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
from mpl_toolkits import mplot3d 
plt.rcParams['axes.grid'] = True
plt.rcParams['figure.max_open_warning'] = 100


def gen_pdf(output_path):
    print(output_path)
	# run pdflatex
    subprocess.run(['pdflatex', output_path.with_suffix(".tex")], check=True, cwd=output_path.parent)
	# delete temp files
    output_path.with_suffix(".aux").unlink()
    output_path.with_suffix(".log").unlink()

def create_subtitle(fig: plt.Figure, grid: SubplotSpec, title: str):
    row = fig.add_subplot(grid)
    row.set_title('\n\n\n'+title, fontweight='medium',fontsize='medium')
    row.set_frame_on(False)
    row.axis('off')


def create_fig(cf_data, cf_name):
    num_of_plots = cf_data["num_of_plots"]
    title = str(cf_data["title"]) + " " + cf_name
    time = cf_data["time"]
    axis_name = cf_data["axis_name"]
    plot_labels = cf_data["plot_labels"]
    len_data = int(len(cf_data["name_data"].keys())/2)
    names = []
    datas = []
    for i in range(len_data):
        names.append(cf_data["name_data"][f"name{i+1}"])
        datas.append(cf_data["name_data"][f"data{i+1}"])
    fig, ax = plt.subplots(num_of_plots, 1, sharex=True)
    plt.rcParams['figure.max_open_warning'] = 800
    fig.tight_layout()
    for i, data in enumerate(datas):
        name = names[i]
        for j, axis in enumerate(name):
            if len(data[axis]) > 0:    
                # add special conditions for special data
                if axis_name[0] == "Thrust":
                    if j < 4:
                        if len(data[axis]) > 0: 
                            # special plot for the thrust to add maxThrust
                            ax[j].plot(time, data["maxThrust"], lw=0.75,label="maxThrust")
                            ax[j].plot(time, data[axis], lw=0.75, label=f"{axis}")
                            ax[j].set_ylabel(plot_labels[j])
                            ax[j].legend()
                elif axis_name[0] == "a_world":
                        if j < 3:
                            ax[j].plot(time, data[axis], lw=0.75, label=f"{axis}")
                            axis_ = axis.replace("stateEstimate.a", "rotated_a")
                            ax[j].plot(time, data[axis_], lw=0.75, label=f"{axis_}")
                            ax[j].set_ylabel(plot_labels[j])
                            ax[j].legend()
                else:
                


                    if len(data[axis]) > 0:
                        if "gyro" in axis_name[0]:
                            data[axis] = np.deg2rad(np.array(data[axis]))
                            ax[j].plot(time, data[axis], lw=0.75, label=f"{axis}")

                        else:
                            ax[j].plot(time, data[axis], lw=0.75, label=f"{axis}")
                        ax[j].set_ylabel(plot_labels[j])
                        ax[0].legend()
        grid = plt.GridSpec(num_of_plots, 1)
        create_subtitle(fig, grid[0, ::], title)
    fig.supxlabel("time [s]",fontsize='small')
    return fig


def loadyaml(file_in):
    with open(file_in, "r") as f:
        file_out = yaml.load(f,Loader=yaml.CSafeLoader)
    return file_out
        

def saveyaml(file_dir, data):
    with open(file_dir + '.yaml', 'w') as f:
        yaml.safe_dump(data, f, default_flow_style=None)

def saveyaml2(file_dir, data):
    with open(file_dir + '.yaml', 'w') as f:
        yaml.safe_dump(data, f, default_flow_style=False)

def flatten(xss):
    return [x for xs in xss for x in xs]

def getData(logDatas, dataname, unit):
    out = []
    for data in dataname:
        if data in logDatas.keys():
            res = np.array(logDatas[data])
            if unit == "mm":
                res = res / 1000.0
            elif unit == "grams":
                res = res / 100.0
            out.extend(res.tolist())
    return out


def extractData(files, start_time=0, end_time=100):
    logDatas = [cfusdlog.decode(f)['fixedFrequency'] for f in files]
    starttime = min([logDatas[k]['timestamp'][0] for k in range(len(files))])
    # filter by time
    for k in range(len(logDatas)):
        t = (logDatas[k]['timestamp'] - starttime)/1000
        idx = np.where(np.logical_and(t > start_time, t < end_time))
        for key, value in logDatas[k].items():
            logDatas[k][key] = value[idx]
    return starttime, logDatas


def computeStats(data, flights):
    ep_trial = []
    energy_trials = []
    trials = 0
    for flight_num, cfs in enumerate(flights):
        flight_data = data[flight_num]
        motorForces = []
        for i, cf in enumerate(cfs):
            for page_key, page_value in flight_data.items():
                cf_data = page_value[cf]
                if cf_data["title"] =="Payload Pose":
                    data_p0 = np.array(list(cf_data["name_data"]["data1"].values())) 
                    data_p0d = np.array(list(cf_data["name_data"]["data2"].values())) 
                    ep_trial.extend(np.linalg.norm(data_p0-data_p0d, axis=0)) 
                elif cf_data["title"] == "Thrust":
                    motor_data = np.array(list(cf_data["name_data"]["data1"].values()))
                    motor_thrust = motor_data[0:4, :]
                    motorForces.append(motor_thrust)
    
        min_size = min(arr.shape[1] for arr in motorForces)
        motorForces_trimmed =  [arr[:, :min_size] for arr in motorForces]

        motorForces_stack = np.concatenate(motorForces_trimmed, axis=0)
        force = np.sum(motorForces_stack, axis=0)/4
        power = force / 4
 
        energy = np.sum(power.tolist())*0.01/60/60 # Wh
        energy_trials.append(energy)
        trials += 1
    
    stats_dict = dict()
    stats_dict["energy_mean"] = np.mean(energy_trials).tolist()
    stats_dict["energy_std"] = np.std(energy_trials).tolist()
    stats_dict["energy_unit"] = "Wh"
    stats_dict["trials"] = trials
    stats_dict["ep_mean"] = dict()
    # stats_dict["ep_mean"]["mean"] = dict()
    # stats_dict["ep_mean"]["std"] = dict()
    stats_dict["ep_mean"]["unit"] = "m"
    flights_flattened = flatten(flights)
    
    stats_dict["ep_mean"]["mean"] = float(np.mean(ep_trial))
    stats_dict["ep_mean"]["std"] = float(np.std(ep_trial))

    return stats_dict

# Special computations are added here:
def computeMotorForces(motor_components, i):
    names =  motor_components[f"name{i+1}"]
    motorpart = []
    for name in names: 
        motorpart.append(np.array([motor_components[f"data{i+1}"][name]]))                   

    motor_components[f"name{i+1}"] = dict()
    motor_components[f"name{i+1}"] = ["f1", "f2", "f3", "f4", "maxThrust"]
    motor_components[f"data{i+1}"] = dict()

    motor_components[f"data{i+1}"]["f1"] = np.array(motorpart[0] - motorpart[1] - motorpart[2] + motorpart[3])[0].tolist()
    motor_components[f"data{i+1}"]["f2"] = np.array(motorpart[0] - motorpart[1] + motorpart[2] - motorpart[3])[0].tolist()
    motor_components[f"data{i+1}"]["f3"] = np.array(motorpart[0] + motorpart[1] + motorpart[2] + motorpart[3])[0].tolist()
    motor_components[f"data{i+1}"]["f4"] = np.array(motorpart[0] + motorpart[1] - motorpart[2] - motorpart[3])[0].tolist()
    motor_components[f"data{i+1}"]["maxThrust"] = np.array(motorpart[4])[0].tolist()
    
    return motor_components

def computeMotorForces_new(motor_components, i):
    names =  motor_components[f"name{i+1}"]
    motorpart = []
    for name in names: 
        motorpart.append(np.array([motor_components[f"data{i+1}"][name]]))                   
    armLength = 0.046
    thrustToTorque = 0.005964552
    arm = 0.707106781 * armLength;
    rollPart  = 0.25 / arm * motorpart[1]
    pitchPart = 0.25 / arm * motorpart[2]
    thrustPart = 0.25 * motorpart[0] 
    yawPart = 0.25 * motorpart[3] / thrustToTorque;
    
    for name in names: 
        motorpart.append(np.array([motor_components[f"data{i+1}"][name]]))                   

    motor_components[f"name{i+1}"] = dict()
    motor_components[f"name{i+1}"] = ["f1", "f2", "f3", "f4"]
    motor_components[f"data{i+1}"] = dict()

    motor_components[f"data{i+1}"]["f1"] = np.array(thrustPart - rollPart - pitchPart + yawPart)[0].tolist()
    motor_components[f"data{i+1}"]["f2"] = np.array(thrustPart - rollPart + pitchPart - yawPart)[0].tolist()
    motor_components[f"data{i+1}"]["f3"] = np.array(thrustPart + rollPart + pitchPart + yawPart)[0].tolist()
    motor_components[f"data{i+1}"]["f4"] = np.array(thrustPart + rollPart - pitchPart - yawPart)[0].tolist()
    return motor_components

def computerpy(quat, i, axis_name):
    names = quat[f"name{i+1}"]
    quats = []
    for name in names:
        quats.append(quat[f"data{i+1}"][name])
    quats = np.array(quats).T
    if len(quats) == 0:
        print(f"Warning!, {axis_name} is empty")
    rpy = rn.to_euler(quats, convention="xyz")
    quat[f"name{i+1}"] = dict()
    quat[f"name{i+1}"] = ['roll', 'pitch', 'yaw']
    quat[f"data{i+1}"] = dict()
    quat[f"data{i+1}"]["roll"]  = rpy[:,0].tolist() 
    quat[f"data{i+1}"]["pitch"]  = rpy[:,1].tolist()
    quat[f"data{i+1}"]["yaw"]  = rpy[:,2].tolist()

    return quat


def computeacc(acc, i):
    names = acc[f"name{i+1}"]
    data = []
    for name in names:
        data.append(acc[f"data{i+1}"][name])
    accb = np.array(data[0:3]).T
    quat = np.array(data[3:7]).T
    acc_calc_world = rn.rotate(quat, accb)
    acc[f"name{i+1}"] = dict()
    acc[f"name{i+1}"] = ['ax_w', 'ay_w', 'az_w']
    acc[f"data{i+1}"] = dict()

    acc[f"data{i+1}"]["ax_w"] = (acc_calc_world[:,0]*9.81).tolist()
    acc[f"data{i+1}"]["ay_w"] = (acc_calc_world[:,1]*9.81).tolist()
    acc[f"data{i+1}"]["az_w"] = ((acc_calc_world[:,2] - np.ones_like(acc_calc_world[:,2]))*9.81).tolist()
    return acc
    
def forcesfromrpm(rpm):
    forces = np.zeros(rpm.shape)
    for k, force in enumerate(forces):
        kw = 4.310657321921365e-08
        force_in_newton = (kw * rpm[k]**2 / 1000) * 9.81
        forces[k, :] = force_in_newton
    return forces

def forcesfrompwm(pwm):
    forces = np.zeros(pwm.shape)
    for k, force in enumerate(forces):
        forces_in_grams = -5.360718677769569 + pwm[k] * 0.0005492858445116151 
        forces[k,:] = (forces_in_grams / 1000) * 9.81
    return forces

def computeFa(aw, q, u, axis_name):
    if "pwm" in axis_name:
        motor_forces_newton = forcesfrompwm(u.T)
    elif "rpm" in axis_name:
        motor_forces_newton = forcesfromrpm(u.T)
    else: 
        print("Wrong input name")
        exit()
    m = 0.038
    g = np.array([0,0,-9.81])
    arm_length = 0.046  # m
    arm = 0.707106781 * arm_length
    t2t = 0.006  # thrust-to-torque ratio
    B0 = np.array([
        [1, 1, 1, 1],
        [-arm, -arm, arm, arm],
        [-arm, arm, arm, -arm],
        [-t2t, t2t, -t2t, t2t]
        ])
    fa = np.zeros(aw.shape)
    # print(motor_forces_newton)
    # exit()
    for k, f in enumerate(fa):
        eta = np.dot(B0, motor_forces_newton[k,:])
        f_u = np.array([0, 0, eta[0]])
        fa[k] = m * aw[k] - rn.rotate(q[:,k], f_u)
    return fa


def computeResidual(states, i, axis_name):
    names = states[f"name{i+1}"]
    print(names)
    data = []
    for name in names:
        data.append(states[f"data{i+1}"][name])
    accb = np.array(data[0:3])
    quat = np.array(data[3:7])
    u    = np.array(data[7:11])

    aw = rn.rotate(quat.T, accb.T)
    aw[:] *= 9.81

    fa = computeFa(aw, quat, u, axis_name)
    states[f"name{i+1}"] = dict()
    states[f"name{i+1}"] = ['Fax_'+axis_name, 'Fay_'+axis_name, 'Faz_'+axis_name]
    states[f"data{i+1}"] = dict()

    states[f"data{i+1}"]["Fax_"+axis_name] = fa[:,0].tolist()
    states[f"data{i+1}"]["Fay_"+axis_name] = fa[:,1].tolist()
    states[f"data{i+1}"]["Faz_"+axis_name] = fa[:,2].tolist()
    return states


def main():
    out = loadyaml("config.yaml")
    # print(out)

if __name__=="__main__":
    main()