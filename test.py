import numpy as np
import matplotlib.pyplot as plt

import utils
from hopper_2d_viz import hopper_plot

from algorithms import RRT, RGRRT, R3T
from models import Pendulum, Hopper1D, Hopper2D

import matplotlib.style as mplstyle
mplstyle.use(['dark_background', 'ggplot', 'fast'])


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


def test_calc_input(ax):
    p = Pendulum(dt=.0001)
    tau=0.1
    x0 = np.array([3,1])

    l = p.get_reachable_AH(x0, tau, convex_hull=True)
    kp = l[0][0]
    AH = l[0][1]

    ax.scatter(x0[0], x0[1], color="red")
    ax.scatter(kp[0],kp[1], color="yellow")
    utils.visualize_polytope_convexhull(AH, x0, ax=ax)

    x_rand = np.array([4,0.7])

    delta = utils.distance_point_polytope(AH=AH, query=x_rand)

    point = x_rand+delta

    ax.scatter(x_rand[0],x_rand[1],marker="x",color="red")
    ax.scatter(point[0],point[1],color="green")

    states, controls = p.calc_input(frm=x0, to=point, dt=tau)
    
    ax.plot(states[:,0],states[:,1],color="purple")

    error = (states[-1]-point)
    print(error)

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

def test_rrt_pendulum(seed=None, ax=None):
    if seed is  None: seed = np.random.randint(0,10**6)
    np.random.seed(seed)
    p = Pendulum(dt=0.001)

    planner = RRT(p, 0.2)

    goal, plan = planner.plan(max_nodes=1000, plt=None)
    print(goal)
    if ax is not None:
        utils.plot(planner.nodes(), ax, plot_all=True)
        if plan is not None: utils.plot (plan, ax, plot_all=True, last_color="lime", int_color="olive")
    print(seed)
    plt.show()


def test_rgrrt_pendulum(seed=None,ax=None):
    if seed is  None: seed = np.random.randint(0,10**6)
    np.random.seed(seed)
    p = Pendulum(dt=0.01)

    ax.scatter(p.initial_state[0],p.initial_state[1], color="purple")
    for goal_state in p.goal_states:
        ax.scatter(goal_state[0],goal_state[1], color="orange",marker="x")

    planner = RGRRT(p, 0.2)

    goal, plan = planner.plan(max_nodes=1000, ax=None)
    print(goal)
    if ax is not None:
        utils.plot(planner.nodes(), ax, plot_all=True)
        if plan is not None: utils.plot (plan, ax, plot_all=True, last_color="lime", int_color="olive")
    print(seed)
    plt.show()

def test_rgrrt_hopper_1d(seed=None,ax=None):
    if seed is  None: seed = np.random.randint(0,10**6)
    np.random.seed(seed)
    h = Hopper1D(dt=0.01, eps_goal=0.05)

    planner = RGRRT(h, 0.04)

    goal, plan = planner.plan(max_nodes=1000, ax=None)
    print(goal)
    if ax is not None:
        utils.plot(planner.nodes(), ax, plot_all=True)
        if plan is not None: utils.plot (plan, ax, plot_all=True, last_color="lime", int_color="olive")
    print(seed)
    plt.show()

def test_r3t_pendulum(seed=None,ax=None):
    if seed is  None: seed = np.random.randint(0,10**6)
    np.random.seed(seed)
    p = Pendulum(dt=0.05, eps_goal=0.05)

    ax.scatter(p.initial_state[0],p.initial_state[1], color="purple")
    for goal_state in p.goal_states:
        ax.scatter(goal_state[0],goal_state[1], color="orange",marker="x")

    planner = R3T(p, 0.2, convex_hull=False)

    goal, plan = planner.plan(max_nodes=1000, ax=None)
    print(goal)
    if ax is not None:
        utils.plot(planner.nodes(), ax, plot_all=True)
        if plan is not None: utils.plot (plan, ax, plot_all=True, last_color="lime", int_color="olive")
    print(seed)
    plt.show()

def test_r3t_hopper_1d(seed=None,ax=None):
    if seed is  None: seed = np.random.randint(0,10**6)
    np.random.seed(seed)
    p = Hopper2D(dt=0.001)

    planner = R3T(p, 0.04)

    goal, plan = planner.plan(max_nodes=1000, ax=None)
    print(goal)
    if ax is not None:
        utils.plot(planner.nodes(), ax, plot_all=True)
        if plan is not None: utils.plot (plan, ax, plot_all=True, last_color="lime", int_color="olive")
    print(seed)
    plt.show()

fig,ax = plt.subplots()
test_r3t_hopper_1d(ax=ax)
#test_calc_input(ax)