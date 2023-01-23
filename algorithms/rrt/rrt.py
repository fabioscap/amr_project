from rtree import index
import numpy as np

class StateTree:
    def __init__(self, 
                 dim, # dimension of the state
                 ):
        self.dim = dim

        # set properties for nearest neighbor
        self.state_tree_p = index.Property()
        self.state_tree_p.dimension = dim
        
        self.state_idx = index.Index(properties=self.state_tree_p)

        # map
        self.state_id_to_state = {}

    def insert(self, state_id, state):
        self.state_idx.insert(state_id, np.concatenate([state,state]))
        self.state_id_to_state[state_id] = state

    def nearest(self, state):
        # https://github.com/wualbert/r3t/blob/master/common/basic_rrt.py
        # they stack the state for some reason

        nearest_id = list(self.state_idx.nearest(state, num_results=1))[0]

        nearest = self.state_id_to_state[nearest_id]

        return nearest

class Node:
    def __init__(self, state, u=None, parent=None, cost=0.):
        self.state = state
        self.u = u
        self.parent = parent
        self.cost = cost
        self.children = set()

    def add_child(self, child):
        self.children.add(child)

    def __hash__(self) -> int:
        return hash(str(self.state))
    
    def __eq__(self, __o: object) -> bool:
        return self.__hash__() == __o.__hash__()

class RRT:
    
    def __init__(self, initial_state, goal_state, eps, state_bounds, extend_func):
        self.dim = initial_state.shape[0]
        self.initial_state = initial_state
        self.goal_state = goal_state
        self.eps = eps
        self.state_bounds = state_bounds # shape(dim,2)

        self.initial_node = Node(initial_state)

        self.state_tree = StateTree(self.dim)

        self.extend_func = extend_func

        # map
        self.state_to_node = {}
        self.initial_state_id = hash(str(self.initial_state))
        self.state_to_node[self.initial_state_id] = self.initial_node

        self.state_tree.insert(self.initial_state_id, self.initial_state)

    # sample q_rand uniformly
    def sample_state(self):
        rnd = np.random.rand(self.dim)

        # normalize in bounds [0,1]->[low,high]
        for i in range(self.dim):
            high = self.state_bounds[i,1]
            low  = self.state_bounds[i,0]
            rnd[i] = (high-low)*rnd[i] + low
        
        return rnd
    
    # find q_near
    def nearest_neighbor(self, q_rand):
        q_near = self.state_tree.nearest(q_rand)

        return q_near
    
    def expand(self, q_near, q_rand):
        q_next, u = self.extend_func(q_near, q_rand)

        node_near = self.state_to_node[hash(str(q_near))]

        # add node to tree
        node_next = Node(q_next, u, node_near, cost=np.linalg.norm(u))

        node_near.add_child(node_next)

        # add state to database
        state_id = hash(str(q_next))
        self.state_tree.insert(state_id, q_next)

        # add link to node
        self.state_to_node[state_id] = node_next

        return node_next



