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
    seed = np.random.randint(0, 10**6)
    seed = 878319
    np.random.seed(seed)
    print(seed)
    p = Pendulum(dt=0.05)

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
                  tau=0.2, rewire=True)
    success, goal_node, nodes = planner.plan(max_nodes=300,plt=None)
    print("nodes", nodes)
    print("polytopes",len(planner.polytope_tree.polytope_id_to_polytope.values()))
    elapsed = time.time()-start
    utils.plot(planner.initial_node, plt=plt)

    """
    for polytope_id in planner.polytope_tree.polytope_id_to_polytope.keys():
        x = planner.polytope_id_to_node[polytope_id].state
        polytope = planner.polytope_tree.polytope_id_to_polytope[polytope_id]
        utils.visualize_polytope_convexhull(polytope,x,plt=plt)
    """
        
    if success:
        plan = planner.get_plan(goal_node, plt=plt)
        plt.scatter(goal_node.state[0], goal_node.state[1], marker="x", c="green")
        print("COST", planner.cumulative_cost(goal_node))
        print(seed)
        pass

    

    print(f"{elapsed} seconds")
    print(f"expanded {nodes} nodes")
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

def test_rgrrt_hopper_1d():
    start = time.time()
     #np.random.seed(1834913)
    p = Hopper1D(dt=0.001)

    q0 = np.array([2,0])
    q_goal = np.array([3, 0])

    plt.scatter(q0[0], q0[1], c="red")
    plt.scatter(q_goal[0], q_goal[1], marker="x", c="red")

    pi = np.pi
    state_bounds = np.zeros((2,2))
    #0.5 5.5
    state_bounds[0] = np.array([0.5,5.5])
    state_bounds[1] = np.array([-10,10])

    planner = RGRRT(q0, q_goal,  0.05, state_bounds, get_reachable_func=p.get_reachable_points,tau=0.04,)
    try:

        success, goal_node, nodes = planner.plan(max_nodes=10000,plt=None)
        elapsed = time.time()-start
        utils.plot(planner.initial_node, plt=plt)
        
        print("nodes", nodes)
        if success:
            plan = planner.get_plan(goal_node, plt=plt)
            plt.scatter(goal_node.state[0], goal_node.state[1], marker="x", c="green")
            print(plan)

        elapsed = time.time()-start
        print(f"{elapsed} seconds")
        print(f"expanded {nodes} nodes")
            
    except KeyboardInterrupt:
        pass


    elapsed = time.time()-start
    utils.plot(planner.initial_node, plt=plt)

    """
    for polytope_id in planner.polytope_tree.polytope_id_to_polytope.keys():
        x = planner.polytope_id_to_node[polytope_id].state
        polytope = planner.polytope_tree.polytope_id_to_polytope[polytope_id]
        utils.visualize_polytope_convexhull(polytope,x,plt=plt)
    """
        

    plt.show()

def test_r3t_hopper_1d():

    seed = np.random.randint(0, 10**6)
    
    np.random.seed(seed)
    print(seed)
    p = Hopper1D(dt=0.001)

    q0 = np.array([2,0])
    q_goal = np.array([3, 0])

    plt.scatter(q0[0], q0[1], s=5, c="red")
    plt.scatter(q_goal[0], q_goal[1], s=5,marker="x", c="red")

    state_bounds = np.zeros((2,2))
    #0.5 5.5
    state_bounds[0] = np.array([0.5,5.5])
    state_bounds[1] = np.array([-10,10])

    planner = R3T(q0, q_goal,  0.1, state_bounds, 
                  solve_input_func=p.calc_input,
                  get_kpoints_func=p.get_reachable_points, 
                  get_polytope_func=p.get_reachable_AH,
                  tau=0.01,rewire=False)
    start = time.time() 
    try:

        success, goal_node, nodes = planner.plan(max_nodes=800,plt=None)
        print("COST", planner.cumulative_cost(goal_node))
        elapsed = time.time()-start
        utils.plot(planner.initial_node, size=3, lw=1, plt=plt)
        
        print("nodes", nodes)
        print("polytopes",len(planner.polytope_tree.polytope_id_to_polytope.values()))
        if success:
            goal_node.states.reverse()
            plan = planner.get_plan(goal_node, plt=plt)
            #plt.scatter(goal_node.state[0], goal_node.state[1],s = 5, marker="x", c="green")
            #utils.plot(planner.initial_node, plt=plt)
            print("cost=",planner.cumulative_cost(goal_node))

        elapsed = time.time()-start
        print(f"{elapsed} seconds")
        print(f"expanded {nodes} nodes")
        plt.show()
            
    except KeyboardInterrupt:
        pass

def test_hopper_2d():
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

def test_r3t_hopper_2d():
    start = time.time()
    
    # get seed
    seed = np.random.randint(0, 10**6)
    #seed = 861006 
    np.random.seed(seed)
    print(seed)
     #np.random.seed(1834913)
    p = Hopper2D(dt=0.005)

    q0 = np.asarray([0., 1., 0, 0, 1.5, 0, 0., 0., 0., 0., 0.])
    q_goal = np.asarray([6,1.,0.,0.,1.5,0.,0.,0.,0.,0., 0.])

    plt.scatter(q0[0], q0[1], s=5, c="red")
    plt.scatter(q_goal[0], q_goal[1], s=5,marker="x", c="red")

    pi = np.pi
    state_bounds = np.zeros((11,2))
    state_bounds[0] = np.array([-0.5,9.5])
    state_bounds[1] = np.array([2,4])
    state_bounds[2] = np.array([-0.5,0.5])
    state_bounds[3] = np.array([-0.4,0.4])
    state_bounds[4] = np.array([1,9])
    state_bounds[5] = np.array([-3,3])
    state_bounds[6] = np.array([-10,10])
    state_bounds[7] = np.array([-4,4])
    state_bounds[8] = np.array([-4,4])
    state_bounds[9] = np.array([-20,20])

    planner = R3T(q0, q_goal,  0.1, state_bounds, 
                  solve_input_func=p.calc_input,
                  get_kpoints_func=p.get_reachable_points, 
                  get_polytope_func=p.get_reachable_AH,
                  tau=0.04,is_hopper_2d=True, rewire=True)

    success, goal_node, nodes = planner.plan(max_nodes=200,plt=None)
    
    elapsed = time.time()-start
    utils.plot_hopper_2d(planner.initial_node, plt=plt)
    
    print("nodes", nodes)
    print("polytopes",len(planner.polytope_tree.polytope_id_to_polytope.values()))
    if success:
        # goal_node.states.reverse()
        plan = planner.get_plan(goal_node, plt=plt, filepath = "./GOAL.txt")
        i = 0
        plt.figure()
        for states in plan:
            for state in states[:-1]:
                X = state[:5]
                if i % 3 ==0:
                    # plot
                    hopper_plot(X,plt, xlim=[-2,17], ylim=[0,5])
                i+= 1
            X = states[-1][:5]
            hopper_plot(X,plt, xlim=[-2,17], ylim=[-0.4,5])
        print(seed)
        print("cost=",planner.cumulative_cost(goal_node))
        #plt.scatter(goal_node.state[0], goal_node.state[1],s = 5, marker="x", c="green")
        #utils.plot(planner.initial_node, plt=plt)

    elapsed = time.time()-start
    print(f"{elapsed} seconds")
    print(f"expanded {nodes} nodes")
    plt.show()

#test_hopper_2d()
#test_rgrrt_hopper_1d()
# test_rrt_pendulum()

#test_point_to_polytope()
# test_AH_to_bbox()

#test_r3t_pendulum()
#test_r3t_hopper_1d()
test_r3t_hopper_2d()
#plot_hopper_1D()