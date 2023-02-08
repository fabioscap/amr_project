import numpy as np
import matplotlib.pyplot as plt
import time
import utils
from hopper_2d_viz import hopper_plot

from algorithms import RRT, RGRRT, R3T
from models import Pendulum, Unicycle, Hopper1D, Hopper2D, Hopper2D_old

from algorithms.r3t.r3t import AABBTree

from rtree import index

def plot_pendulum():
    step_size = 0.01
    p = Pendulum(b=0.1, dt=step_size)

    q = np.array([0.0, 0.0])
    plt.scatter(q[0],q[1],c="red")
    thetas = [q[0]]
    dthetas = [q[1]]
    for i in range(50000):
        q = p.step(q, 0.0)
        thetas.append(q[0])
        dthetas.append(q[1])


    plt.plot(thetas,dthetas)

    plt.xlabel("theta")
    plt.ylabel("dtheta")
    plt.show()

def plot_hopper_1D():
    p = Hopper1D()

    x = np.array([2, 0.0])
    plt.scatter(x[0],x[1],c="red")
    h = [x[0]]
    dh = [x[1]]
    for i in range(5000):
        x = p.step(x, 80)
        h.append(x[0])
        dh.append(x[1])

    plt.plot(h,dh)

    plt.xlabel("h")
    plt.ylabel("dh")
    plt.show()

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
    p = Pendulum(dt=0.02)

    planner = RRT(p, 0.2)

    goal, goal_node, n_nodes = planner.plan(max_nodes=10000, plt=None)
    print(goal)
    
    import matplotlib.style as mplstyle
    mplstyle.use(['dark_background', 'ggplot', 'fast'])
    utils.plot(planner, plt, plot_all=True)
    print(seed)
    plt.show()

def test_rgrrt_pendulum(seed=None):
    if seed is  None: seed = np.random.randint(0,10**6)
    np.random.seed(seed)
    p = Pendulum(dt=0.02)

    planner = RGRRT(p, 0.2)

    goal, goal_node, n_nodes = planner.plan(max_nodes=10000, plt=None)
    print("\n",goal)
    
    import matplotlib.style as mplstyle
    mplstyle.use(['dark_background', 'ggplot', 'fast'])
    utils.plot(planner, plt, plot_all=True)
    print(seed)
    plt.show()

test_rrt_pendulum()
test_rgrrt_pendulum()