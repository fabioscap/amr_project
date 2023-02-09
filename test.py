import numpy as np
import matplotlib.pyplot as plt
import time
import utils
from hopper_2d_viz import hopper_plot

from algorithms import RRT, RGRRT, R3T
from models import Pendulum, Hopper1D, Hopper2D

import pypolycontain as pp


def plot_pendulum(step_size=0.01, ax= None):
    if ax is None:
        fig, ax = plt.subplots()

    p = Pendulum(b=0.1, dt=step_size)

    x = np.array([1.5, 0.0])
    u = np.array([1.0])
    plt.scatter(x[0],x[1],c="red")
    thetas = [x[0]]
    dthetas = [x[1]]

    seconds = 10
    for i in range(int(seconds//step_size)):
        x = p.step(x, u, step_size)
        thetas.append(x[0])
        dthetas.append(x[1])

    ax.plot(thetas,dthetas)
    ax.set_xlabel("theta")
    ax.set_ylabel("dtheta")
    ax.set_ylim([-10,10])


def plot_hopper_1D(step_size=0.01, t=10):
    p = Hopper1D(dt=step_size)

    x = np.array([2, 0.0])
    u = np.array([40.0])
    plt.scatter(x[0],x[1],c="red")
    h = [x[0]]
    dh = [x[1]]
    for i in range(int(t//step_size)):
        if x[1] < 0:
            u = np.array([0.0])
        else:
            u = np.array([40.0]) 
        x = p.step(x, u, step_size)
        h.append(x[0])
        dh.append(x[1])

    plt.plot(h,dh)

    plt.xlabel("h")
    plt.ylabel("dh")


def plot_hopper_2d():
    h = Hopper2D(dt=0.01)
    x = np.array([0,1,0,0,1.5,1,0,0,0,0,0])
    x_ = x.copy()
    u = np.array([0,0])
    plt.figure()
    for i in range(1000):
        x = h.step(x, u)
        if i % 10 == 0:
            hopper_plot(x[:5], plt)
            plt.draw()
            plt.pause(0.1)

        # print(np.linalg.norm(x_j- x_j_))

    plt.show()

def test_rrt_pendulum(seed=None):
    if seed is  None: seed = np.random.randint(0,10**6)
    np.random.seed(seed)
    p = Pendulum(dt=0.001)

    planner = RRT(p, 0.2)

    goal, goal_node, n_nodes = planner.plan(max_nodes=1000, plt=None)
    print(goal)
    
    import matplotlib.style as mplstyle
    mplstyle.use(['dark_background', 'ggplot', 'fast'])
    utils.plot(planner, plt, plot_all=True)
    print(seed)
    plt.show()

def test_rgrrt_pendulum(seed=None):
    if seed is  None: seed = np.random.randint(0,10**6)
    np.random.seed(seed)
    p = Pendulum(dt=0.001)

    planner = RGRRT(p, 0.2)

    goal, goal_node, n_nodes = planner.plan(max_nodes=10000, plt=None)
    print("\n",goal)
    
    import matplotlib.style as mplstyle
    mplstyle.use(['dark_background', 'ggplot', 'fast'])
    utils.plot(planner, plt, plot_all=True)
    print(seed)
    plt.show()

def test_rgrrt_hopper_1d(seed=None):
    if seed is  None: seed = np.random.randint(0,10**6)
    np.random.seed(seed)
    h = Hopper1D(dt=0.001, eps_goal=0.05)

    planner = RGRRT(h, 0.04)

    goal, goal_node, n_nodes = planner.plan(max_nodes=2000, plt=None)
    print("\n",goal)
    print(planner.min_distance)
    import matplotlib.style as mplstyle
    mplstyle.use(['dark_background', 'ggplot', 'fast'])
    utils.plot(planner, plt, plot_all=True)
    print(seed)
    plt.show()

#test_rrt_pendulum()
#test_rgrrt_hopper_1d()