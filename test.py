import numpy as np
import matplotlib.pyplot as plt

from algorithms import RRT
from models import Pendulum

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

def test_rrt():
    p = Pendulum(dt=0.1)
    q0 = np.zeros(2)

    plt.scatter(q0[0], q0[1])

    pi = np.pi
    state_bounds = np.zeros((2,2))

    state_bounds[0] = np.array([-pi,pi])
    state_bounds[1] = np.array([-10,10])


    rrt = RRT(q0, q0, None, state_bounds, p.extend_to)

    for i in range(100):
        q_rand = rrt.sample_state()

        q_near = rrt.nearest_neighbor(q_rand)

        node_next = rrt.expand(q_near, q_rand)

        q_next = node_next.state
        q_parent = node_next.parent.state
        plt.scatter(q_next[0], q_next[1])
        plt.plot([q_parent[0], q_next[0]],[q_parent[1], q_next[1]])


        # plt.draw()
        # plt.pause(0.5)

    plt.show()


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

rrt.plan(max_iters=7000, plt=plt)

plt.show()