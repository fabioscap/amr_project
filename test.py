import numpy as np
import matplotlib.pyplot as plt
import time
import utils

from algorithms import RRT, RGRRT, R3T
from models import Pendulum, Unicycle

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

    p = Pendulum(dt=0.01, l=0.5, g=9.8)
    q0 = np.zeros(2)
    q_goal = np.array([np.pi, 0])

    plt.scatter(q0[0], q0[1], c="red")
    plt.scatter(q_goal[0], q_goal[1], marker="x", c="red")

    pi = np.pi
    state_bounds = np.zeros((2,2))

    state_bounds[0] = np.array([-3*pi/2,3*pi/2])
    state_bounds[1] = np.array([-12,12])


    rrt = RRT(q0, q_goal,  0.05, state_bounds, p.extend_to, tau=p.dt)

    success, goal_node, n_nodes = rrt.plan(max_nodes=100000, plt=None)
    elapsed = time.time()-start
    # utils.plot(rrt.initial_node, plt=plt)

    if success:
        plan = rrt.get_plan(goal_node, plt=plt)
        plt.scatter(goal_node.state[0], goal_node.state[1], marker="x", c="green")
        print("n_nodes", n_nodes)

    print(f"{elapsed} seconds")
    plt.show()

def test_rgrrt_pendulum():
    start = time.time()
    np.random.seed(1834913)
    p = Pendulum(dt=0.01)

    q0 = np.zeros(2)
    q_goal = np.array([np.pi, 0])

    plt.scatter(q0[0], q0[1], c="red")
    plt.scatter(q_goal[0], q_goal[1], marker="x", c="red")

    pi = np.pi
    state_bounds = np.zeros((2,2))

    state_bounds[0] = np.array([-3*pi/2,3*pi/2])
    state_bounds[1] = np.array([-10,10])

    planner = RGRRT(q0, q_goal,  0.05, state_bounds, p.get_reachable_points, tau=0.1)

    success, goal_node, nodes = planner.plan(max_nodes=10000, plt=None)
    
    elapsed = time.time()-start
    utils.plot(planner.initial_node, plt=plt)

    if success:
        plan = planner.get_plan(goal_node, plt=plt)
        plt.scatter(goal_node.state[0], goal_node.state[1], marker="x", c="green")
        print(plan)
        pass

    

    print(f"{elapsed} seconds")
    print(f"expanded {nodes} nodes")
    plt.show()

def test_point_to_polytope():
    import warnings
    warnings.filterwarnings("ignore")
    import pypolycontain as pp

    n_dim = 4
    
    query = np.ones(n_dim)*4

    AH = pp.to_AH_polytope(pp.unitbox(n_dim).H_polytope)

    
    utils.distance_point_polytope(query, AH)


def test_AH_to_bbox():
    import warnings
    warnings.filterwarnings("ignore")
    import pypolycontain as pp

    # rhombus
    H = np.zeros((4,2))
    H[0,0] = 3/4
    H[0,1] = -1
    H[1,0] = -3/4
    H[1,1] = 1
    H[2,0] = 3/4
    H[2,1] = 1
    H[3,0] = -3/4
    H[3,1] = -1

    h = np.ones(4)*3
    
    P = pp.H_polytope(H,h)
    AH = pp.to_AH_polytope(P)

    bbox = utils.AABB.from_AH(AH)


def test_bbox_tree():
    tree = AABBTree(3)
    import pypolycontain as pp


    l1 = np.array([-1,-1,-1])
    u1 = np.array([1,1,1])

    bbox1 = utils.AABB(l1,u1)

    l2 = l1 + 1
    u2 = u1 + 1
    bbox2 = utils.AABB(l2,u2)

    tree.insert_bbox(hash(bbox1), bbox1)
    tree.insert_bbox(hash(bbox2), bbox2)

    l3 = np.array([1.2,1.2,1.2])
    u3 = np.array([1.4,1.4,1.4])

    bbox3 = utils.AABB(l3,u3)

    print(tree.intersection(bbox3))

def test_r3t_pendulum():
    start = time.time()
    # np.random.seed(1834913)
    p = Pendulum(dt=0.01)

    q0 = np.zeros(2)
    q_goal = np.array([np.pi, 0])

    plt.scatter(q0[0], q0[1], c="red")
    plt.scatter(q_goal[0], q_goal[1], marker="x", c="red")

    pi = np.pi
    state_bounds = np.zeros((2,2))

    state_bounds[0] = np.array([-3*pi/2,3*pi/2])
    state_bounds[1] = np.array([-10,10])

    planner = R3T(q0, q_goal,  0.05, state_bounds, 
                  solve_input_func=p.calc_input,
                  get_kpoints_func=p.get_reachable_points, 
                  get_polytope_func=p.get_reachable_AH,
                  tau=0.2,)
    success, goal_node, nodes = planner.plan(max_nodes=3000,plt=None)
    print("nodes", nodes)
    print("polytopes",len(planner.polytope_tree.polytope_id_to_polytope.values()))
    elapsed = time.time()-start
    utils.plot(planner.initial_node, plt=plt)
    for polytope_id in planner.polytope_tree.polytope_id_to_polytope.keys():
        x = planner.polytope_id_to_node[polytope_id].state
        polytope = planner.polytope_tree.polytope_id_to_polytope[polytope_id]
        utils.visualize_polytope_convexhull(polytope,x,plt=plt)

    if success:
        plan = planner.get_plan(goal_node, plt=plt)
        plt.scatter(goal_node.state[0], goal_node.state[1], marker="x", c="green")
        print(plan)
        pass

    

    print(f"{elapsed} seconds")
    print(f"expanded {nodes} nodes")
    plt.show()


#test_rgrrt_pendulum()
# test_rrt_pendulum()

#test_point_to_polytope()
# test_AH_to_bbox()

test_r3t_pendulum()
