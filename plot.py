import pickle, utils, os, random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.style as mplstyle

def plot(picklename):
    mplstyle.use(['ggplot', 'fast'])
    with open(picklename, "rb") as plan_file:
        dictionary = pickle.load(plan_file)

    if dictionary["model_name"] == "pendulum":
        plot_pendulum(dictionary)
    elif dictionary["model_name"] == "hopper1d":
        plot_hopper1d(dictionary)
    elif dictionary["model_name"] == "hopper2d":
        plot_hopper2d(dictionary)
    else:
        raise Exception()
    return


def plot_pendulum(d):

    dir = os.getcwd()
    dir += "/trajectories/"

    folder_name = f"{d['model_name']}_{d['planner_name']}_{d['seed']}"
    if os.path.exists(dir+folder_name):
        name_rnd = '_v'+str(random.randint(0, 100))
        folder_name += name_rnd
    os.mkdir(dir+folder_name)

    fig, ax = plt.subplots()

    planner = d["planner"]
    plan    = d["plan"]

    ax.set_title( f"{d['model_name']} {d['planner_name']} nodes: {d['planner'].n_nodes}")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("xdot [m/s]")
    utils.plot(planner.nodes(), ax, plot_all=True, polytopes=False)
    if plan is not None: utils.plot (plan, ax, plot_all=True, last_color="lime", int_color="olive", polytopes=False, lw=3)
    plt.savefig(dir+folder_name+"/nodes.png")
    plt.close()

    plan = d["plan"]
    if plan == None:
        return

    states = []
    controls = []
    for node in plan:
        for i in range(node.states.shape[0]):
            states.append(node.states[i,:])
            controls.append(node.controls[i,:])


    T = np.arange(0, len(states)) * d["model"].dt

    states   = np.array(states)
    controls = np.array(controls)

    fig, (ax_x, ax_dx) = plt.subplots(2,1)
    
    ax_x.plot(T, states[:,0])
    ax_dx.plot(T, states[:,1])

    ax_x.set_xlabel("t [s]")
    ax_x.set_ylabel("x [m]")
    ax_dx.set_xlabel("t [s]")
    ax_dx.set_ylabel("xdot [m/s]")
    plt.savefig(dir+folder_name+"/traj.png")
    plt.close()

    fig, (ax_c) = plt.subplots()
    ax_c.plot(T, controls)
    ax_c.set_xlabel("t [s]")
    ax_c.set_ylabel("F [N]")
    plt.savefig(dir+folder_name+"/inputs.png")
    plt.close()

    return

def plot_hopper2d(d):

    return

def plot_hopper1d(d):

    dir = os.getcwd()
    dir += "/trajectories/"

    folder_name = f"{d['model_name']}_{d['planner_name']}_{d['seed']}"
    if os.path.exists(dir+folder_name):
        name_rnd = '_v'+str(random.randint(0, 100))
        folder_name += name_rnd
    os.mkdir(dir+folder_name)

    fig, ax = plt.subplots()

    planner = d["planner"]
    plan    = d["plan"]

    ax.set_title( f"{d['model_name']} {d['planner_name']} nodes: {d['planner'].n_nodes}")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("xdot [m/s]")
    utils.plot(planner.nodes(), ax, plot_all=True, polytopes=False)
    if plan is not None: utils.plot (plan, ax, plot_all=True, last_color="lime", int_color="olive", polytopes=False, lw=3)
    plt.savefig(dir+folder_name+"/nodes.png")
    plt.close()

    plan = d["plan"]
    if plan == None:
        return

    states = []
    controls = []
    for node in plan:
        for i in range(node.states.shape[0]):
            states.append(node.states[i,:])
            controls.append(node.controls[i,:])


    T = np.arange(0, len(states)) * d["model"].dt

    states   = np.array(states)
    controls = np.array(controls)

    fig, (ax_x, ax_dx) = plt.subplots(2,1)
    
    ax_x.plot(T, states[:,0])
    ax_dx.plot(T, states[:,1])

    ax_x.set_xlabel("t [s]")
    ax_x.set_ylabel("x [m]")
    ax_dx.set_xlabel("t [s]")
    ax_dx.set_ylabel("xdot [m/s]")
    plt.savefig(dir+folder_name+"/traj.png")
    plt.close()

    fig, (ax_c) = plt.subplots()
    ax_c.plot(T, controls)
    ax_c.set_xlabel("t [s]")
    ax_c.set_ylabel("F [N]")
    plt.savefig(dir+folder_name+"/inputs.png")
    plt.close()

    return


if __name__ == "__main__":
    picklename="pendulum_R3T_602464.pickle"
    plot(picklename)

