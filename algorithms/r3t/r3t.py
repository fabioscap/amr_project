from algorithms.planner import Planner,StateTree,PolytopeTree,Node
import numpy as np
from models.model import Model
import utils
import matplotlib.pyplot as plt

class R3T(Planner):
    
    def __init__(self, model: Model, tau, thr=1e-5, convex_hull=True):
        super().__init__(model, tau, thr)

        self.convex_hull = convex_hull

        # we need a polytope tree to find the nearest polytope given a query point
        self.polytope_tree = PolytopeTree(model.x_dim)

        self.polytope_id_to_node = {}

    def add_node(self, states, controls = None, cost=None, parent:Node=None):
        if len(states.shape) == 1:
            # if it's just a single state then reshape it to still be a collection of states
            states = states.reshape(1,-1)
        if controls is None:
            cost = 0
            controls = np.empty((1,self.u_dim))

        if cost is None:
            cost = np.sum( np.linalg.norm(controls) ) # sum the cost of every control
        node = Node(states, controls, parent, cost, self.model.dt)

        # manage the parent's children
        if parent is not None:
            is_new = parent.add_child(node)
            assert is_new
        
        self.n_nodes += 1

        state_id = self.state_tree.insert(states[-1])
        self.id_to_node[state_id] = node

        # we need to get polytopes and respective keypoints and put those in the tree
        reachables = self.model.get_reachable_AH(states[-1], self.tau, convex_hull=True)
        node.polytopes = []
        for reachable in reachables:
            kpoint = reachable[0]
            polytope = reachable[1]

            """
            p = utils.visualize_polytope_convexhull(polytope, states[-1], ax=ax)
            plt.draw()
            plt.pause(0.01)
            """

            self.polytope_tree.insert(polytope, kpoint)
            self.polytope_id_to_node[hash(polytope)] = node

            node.polytopes.append(polytope)

        return node
    
    def expand(self, x_rand):

        # get the nearest reachable point
        polytope, point, dist = self.polytope_tree.nearest_polytope(x_rand)

        polytope_id = hash(polytope)
        node_near = self.polytope_id_to_node[polytope_id]

        x_near = node_near.state

        states, controls = self.model.calc_input(frm=x_near, to=point, dt=self.tau)
        
        states = np.vstack(( states , self.model.ffw(states[-1]) ))

        x_next = states[-1]

        closest_idx = self.state_tree.nearest(x_next)
        closest_state = self.id_to_node[closest_idx].state


        if np.linalg.norm(x_next - closest_state) < self.thr:
            # there is already a node at this location
            # TODO consider rewiring if the cost is less
            return None, None
        
        cost = np.sum( np.linalg.norm(controls) )

        # add node to tree
        node_next = self.add_node(states, controls, cost, node_near)

        return node_next, node_near