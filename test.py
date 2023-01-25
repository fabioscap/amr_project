import numpy as np
import matplotlib.pyplot as plt
import time

from algorithms import RRT, RGRRT
from models import Pendulum, Unicycle

from rtree import index

def plot_pendulum():
    p = Pendulum(m_l=0.5, b=1)

    q = np.array([0.0, 0.5])
    plt.scatter(q[0],q[1],c="red")
    thetas = [q[0]]
    dthetas = [q[1]]
    for i in range(50000):
        q = p.step(q, 0.0, 0.001)
        thetas.append(q[0])
        dthetas.append(q[1])


    plt.plot(thetas,dthetas)

    plt.xlabel("theta")
    plt.ylabel("dtheta")
    plt.show()


def test_rtree():
    n = 100
    points = np.random.rand(n, 2)

    query = np.random.rand(2)

    state_tree_p = index.Property()
    state_tree_p.dimension = 2
            
    state_idx = index.Index(properties=state_tree_p)

    for p in range(n):
        state_idx.insert(p, points[p])

    matches = state_idx.nearest(query, num_results=2)

    nearest = points[list(matches)[0]]

    # points
    plt.scatter(points[:,0],points[:,1],c="blue")

    # query
    plt.scatter(query[0], query[1], c="red")

    # match 
    plt.scatter(nearest[0], nearest[1], c="green")

    plt.show()

def test_rrt_pendulum():
    start = time.time()
    np.random.seed(1834913)
    p = Pendulum(m_l=0.5 ,dt=0.1)
    q0 = np.zeros(2)
    q_goal = np.array([np.pi, 0])

    plt.scatter(q0[0], q0[1], c="red")
    plt.scatter(q_goal[0], q_goal[1], marker="x", c="red")

    pi = np.pi
    state_bounds = np.zeros((2,2))

    state_bounds[0] = np.array([-3*pi/2,3*pi/2])
    state_bounds[1] = np.array([-10,10])


    rrt = RRT(q0, q_goal,  0.05, state_bounds, p.extend_to)

    success, goal_node = rrt.plan(max_iters=20000, plt=None)
    elapsed = time.time()-start
    if success:
        plan = rrt.get_plan(goal_node, plt=plt)
        plt.scatter(goal_node.state[0], goal_node.state[1], marker="x", c="green")
        # print(plan)
    print(f"{elapsed} seconds")
    plt.show()

def test_rgrrt_pendulum():
    start = time.time()
    # np.random.seed(1834913)
    p = Pendulum(m_l=0.5 ,dt=0.01)

    q0 = np.zeros(2)
    q_goal = np.array([np.pi, 0])

    plt.scatter(q0[0], q0[1], c="red")
    plt.scatter(q_goal[0], q_goal[1], marker="x", c="red")

    pi = np.pi
    state_bounds = np.zeros((2,2))

    state_bounds[0] = np.array([-3*pi/2,3*pi/2])
    state_bounds[1] = np.array([-10,10])

    planner = RGRRT(q0, q_goal,  0.1, state_bounds, p.get_reachable_points)

    success, goal_node, nodes = planner.plan(max_nodes=80000, plt=None)
    elapsed = time.time()-start
    if success:
        plan = planner.get_plan(goal_node, plt=plt)
        plt.scatter(goal_node.state[0], goal_node.state[1], marker="x", c="green")
        #print(plan)
    print(f"{elapsed} seconds")
    print(f"expanded {nodes} nodes")
    plt.show()

def test_rgrrt_car():
    start = time.time()
    # np.random.seed(1834913)
    p = Pendulum(m_l=0.5 ,dt=0.1)

    q0 = np.zeros(2)
    q_goal = np.array([np.pi/2, -0.2])

    plt.scatter(q0[0], q0[1], c="red")
    plt.scatter(q_goal[0], q_goal[1], marker="x", c="red")

    pi = np.pi
    state_bounds = np.zeros((2,2))

    state_bounds[0] = np.array([-3*pi/2,3*pi/2])
    state_bounds[1] = np.array([-10,10])

    planner = RGRRT(q0, q_goal,  0.2, state_bounds, p.get_reachable_points)

    success, goal_node, nodes = planner.plan(max_nodes=20000, plt=None)
    elapsed = time.time()-start
    if success:
        plan = planner.get_plan(goal_node, plt=plt)
        plt.scatter(goal_node.state[0], goal_node.state[1], marker="x", c="green")
        #print(plan)
    print(f"{elapsed} seconds")
    print(f"expanded {nodes} nodes")
    plt.show()    

test_rgrrt_pendulum()
# test_rrt_pendulum()