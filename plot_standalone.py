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

def create_subtitle(fig: plt.Figure, grid: SubplotSpec, title: str):
    row = fig.add_subplot(grid)
    row.set_title('\n\n\n'+title, fontweight='medium',fontsize='medium')
    row.set_frame_on(False)
    row.axis('off')

def create_fig(data,num_of_plots=3):
    fig, ax = plt.subplots(num_of_plots, 1, sharex=True)
    grid = plt.GridSpec(num_of_plots, 1)
    time = data["time"]
    acc_w = data_to_plot["acc_w"] 
    acc_rotated = data_to_plot["acc_rotated"] 
    acc = data_to_plot["acc"] 

    # ax[0].plot(time, acc[0,:], lw=0.75,label="acc_b")
    ax[0].plot(time, acc_w[0,:], lw=0.75,label="acc_w")
    ax[0].plot(time, acc_rotated[0,:], lw=0.75,label="acc_rotated")

    # ax[1].plot(time, acc[1,:], lw=0.75,label="acc_b")
    ax[1].plot(time, acc_w[1,:], lw=0.75,label="acc_w")
    ax[1].plot(time, acc_rotated[1,:], lw=0.75,label="acc_rotated")

    # ax[2].plot(time, acc[2,:], lw=0.75,label="acc_b")
    ax[2].plot(time, acc_w[2,:], lw=0.75,label="acc_w")
    ax[2].plot(time, acc_rotated[2,:], lw=0.75,label="acc_rotated")

    ax[0].legend()
    create_subtitle(fig, grid[0, ::], "acc")
    fig.supxlabel("time [s]",fontsize='small')
    return fig





f = "data/test_data/cf3_t_07"
data  = cfusdlog.decode(f)['fixedFrequency'] 
starttime = data['timestamp'][0] 
time = ((data['timestamp'] - starttime)/1000.0).tolist()

print(starttime)

rotType = "quat" # or "rpy"
for key in data.keys():
    if "acc.x" in key:
        print("acc in body done")
        acc = np.array([data["acc.x"] ,data["acc.y"], data["acc.z"]])
    elif "stateEstimate.ax" in key:
        print("acc in world done")
        acc_w = np.array([data["stateEstimate.ax"] ,data["stateEstimate.ay"], data["stateEstimate.az"]])
    elif "stateEstimate.qw" in key:
        print("quat done, rot type: "+ str(rotType))
        rotType = "quat"
        quat = np.array([data["stateEstimate.qw"], data["stateEstimate.qx"] ,data["stateEstimate.qy"], data["stateEstimate.qy"]])
    elif "ctrlLeeP.rpyx" in key:
        rotType = "rpy"
        print("rpy done, rot type: " + str(rotType))
        rpy = np.array([data["ctrlLeeP.rpyx"] ,data["ctrlLeeP.rpyy"], data["ctrlLeeP.rpyy"]])
    else:
        continue


if rotType == "quat":
    acc_rotated = rn.rotate(quat.T, acc.T)
elif rotType == "rpy":
    quat = rn.from_euler(rpy[0], rpy[1], rpy[2], convention="xyz", axis_type="extrinsic")
    acc_rotated = rn.rotate(quat, acc.T)



result_pdf = PdfPages(f'result.pdf')
data_to_plot = dict()
data_to_plot["time"] = time
data_to_plot["acc_w"] = acc_w
data_to_plot["acc_rotated"] = acc_rotated.T
data_to_plot["acc"] = acc

fig = create_fig(data_to_plot)
fig.savefig(result_pdf, format='pdf', bbox_inches='tight')
result_pdf.close()