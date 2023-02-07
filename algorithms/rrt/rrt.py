from rtree import index
import numpy as np
from models.model import Model
import time

class StateTree:
    def __init__(self, dim):
        self.dim = dim

        # set properties for nearest neighbor
        self.state_tree_p = index.Property()
        self.state_tree_p.dimension = dim
        
        self.state_idx = index.Index(properties=self.state_tree_p)

        # map
        self.state_id_to_state = {}

    def insert(self, state_id, state):
        self.state_idx.insert(state_id, state)
        self.state_id_to_state[state_id] = state

    def nearest(self, state):
        nearest_id = list(self.state_idx.nearest(state, num_results=1))[0]

        nearest = self.state_id_to_state[nearest_id]

        return nearest_id, nearest

class Node:
    def __init__(self, states:np.ndarray, # an array of dimension   (n, dim_x)
                       controls:np.ndarray, # an array of dimension (m, dim_u)
                       parent=None, 
                       cost=0.):

        self.states = states
        self.controls = controls

        self.state = self.states[-1,:] 

        self.parent:Node = parent

        self.cost = cost
        self.children = set()

    def cumulative_cost(self):
        cost = self.cost
        while self.parent is not None:
            cost += self.parent.cost
            return cost + self.parent.cumulative_cost()
        
    def add_child(self, child):
        if child in self.children:
            return False
        else:
            self.children.add(child)
            return True

    def __hash__(self) -> int:
        return hash(str(np.hstack((self.states.flatten(), self.controls.flatten()))))
    
    def __ex__(self, __o: object) -> bool:
        return self.__hash__() == __o.__hash__()
    
    def __repr__(self) -> str:
        return f"[{self.state}, {self.u}]"

class RRT:
    
    def __init__(self, model: Model, tau, thr=1e-4):

        self.model = model
        self.x_dim = model.x_dim
        self.u_dim = model.u_dim
        self.state_tree = StateTree(self.x_dim)

        self.tau = tau

        self.min_distance = np.inf

        self.n_nodes = 0

        self.thr = thr # threshold to alias nodes

        self.id_to_node = {}


    def add_node(self, states, controls = None, cost=None, parent:Node=None):
        if len(states.shape) == 1:
            # if it's just a single state then reshape it to still be a collection of states
            states = states.reshape(1,-1)
        if controls is None:
            cost = 0
            controls = np.empty((1,self.u_dim))
        elif len(controls.shape) == 0:
            controls = controls.reshape(1,-1)
        if cost is None:
            cost = np.sum( np.linalg.norm(controls, axis=1) ) # sum the cost of every control
        node = Node(states, controls, parent, cost)

        # manage the parent's children
        if parent is not None:
            is_new = parent.add_child(node)
            assert is_new
        
        self.n_nodes += 1

        state_id = self._id(states[-1])
        self.id_to_node[state_id] = node
        self.state_tree.insert(state_id, states[-1])
        return node

    # find x_near
    def nearest_neighbor(self, x_rand):
        id_near, _ = self.state_tree.nearest(x_rand)

        return self.id_to_node[id_near]
    
    def expand(self, node_near: Node, x_rand: np.ndarray):
        x_near = node_near.state

        states, controls = self.model.expand_toward(x_near, x_rand, self.tau)

        if states is None:
            return None # cannot reach that state
        
        x_next: np.ndarray = states[-1]

        _, closest_state = self.state_tree.nearest(x_next)

        if np.linalg.norm(x_next - closest_state) < self.thr:
            # there is already a node at this location
            # TODO consider rewiring if the cost is less
            return None


        cost = np.sum( np.linalg.norm(controls, axis=1) )

        # add node to tree
        node_next = self.add_node(states, controls, cost, node_near)

        return node_next

    def plan(self, max_nodes):
        # add the first node with the initial state
        initial_state = self.model.initial_state

        initial_node = self.add_node(initial_state)

        goal, distance = self.model.goal_check(self.model.initial_state)
        if distance < self.min_distance:
            self.min_distance = distance
        if goal:
            return True, self.initial_node, self.n_nodes
        
        start = time.time()
        while self.n_nodes < max_nodes:
            t = time.time()
            if self.n_nodes%100 == 0:
                print(f"n_nodes: {self.n_nodes}, d: {self.min_distance}, t: {t-start} sec")

            x_rand = self.model.sample()

            node_near = self.nearest_neighbor(x_rand)

            node_next = self.expand(node_near, x_rand)

            if node_next is None:
                continue

            x_next = node_next.state

            goal, distance = self.model.goal_check(x_next)
            if distance < self.min_distance:
                self.min_distance = distance
            if goal:
                return True, node_next, self.n_nodes

        return False, None, self.n_nodes
    
    def _id(self,x:np. ndarray):
        return hash(str(x))

    def get_plan(self, node: Node):
        # go back until root
        plan = [node]
        while node.parent != None:
            plan = [node.parent] + plan
            node = node.parent
        return plan