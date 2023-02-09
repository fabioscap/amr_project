from models import Model
from rtree import index
import numpy as np

class Node:
    def __init__(self, states:np.ndarray, # an array of dimension   (n, dim_x)
                       u:np.ndarray, # an array of dimension (m, dim_u)
                       parent=None, 
                       cost=0.,
                       dt = None):

        self.states = states
        self.u = u

        self.parent:Node = parent

        self.cost = cost
        self.dt = dt

        self.children = set()

    @property
    def state(self):
        return self.states[-1,:]

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
        return hash(str(np.hstack((self.states.flatten(), self.u.flatten()))))
    
    def __ex__(self, __o: object) -> bool:
        return self.__hash__() == __o.__hash__()
    
    def __repr__(self) -> str:
        return f"[{self.state}, {self.u}]"
    
class Planner:
    def __init__(self, model: Model, tau, thr=1e-9):

        self.model = model
        self.x_dim = model.x_dim
        self.u_dim = model.u_dim
        self.state_tree = StateTree(self.x_dim)
        self.tau = tau
        self.thr = thr # threshold to alias nodes

        self.min_distance = np.inf

        self.n_nodes = 0

        self.id_to_node: dict[int, Node] = {}
        
    def _id(self,x:np.ndarray):
        return hash(str(x))
            
    def nodes(self):
        return self.id_to_node.values()
    
    def get_plan(self, node):
        nodes = [node]

        while node.parent is not None:
            nodes = [node.parent] + nodes
            node = node.parent
        
        return nodes
        


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
