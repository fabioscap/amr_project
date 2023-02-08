from algorithms.rrt.rrt import StateTree, Node
import numpy as np
from models.model import Model
import time

class RGRRT:
    def __init__(self, model: Model, tau, thr=1e-9):

        self.model = model
        self.x_dim = model.x_dim
        self.u_dim = model.u_dim
        self.state_tree = StateTree(self.x_dim)

        # we need an additional tree in which we store sampled reachable points
        self.reachable_tree = StateTree(self.x_dim)

        self.tau = tau

        self.min_distance = np.inf

        self.n_nodes = 0

        self.thr = thr # threshold to alias nodes

        self.id_to_node: dict[int, Node] = {}

        # and an additional map to link the reachable points to their nodes
        # (and the dynamic trajectory that generated them)
        self.r_id_to_node: dict[int, tuple[Node, np.ndarray, np.ndarray]] = {}


    def add_node(self, states, controls = None, cost=None, parent:Node=None):
        if len(states.shape) == 1:
            # if it's just a single state then reshape it to still be a collection of states
            states = states.reshape(1,-1)
        if controls is None:
            cost = 0
            controls = np.empty((1,self.u_dim))

        if cost is None:
            cost = np.linalg.norm(controls) 
        node = Node(states, controls, parent, cost)

        # manage the parent's children
        if parent is not None:
            is_new = parent.add_child(node)
            assert is_new
        
        self.n_nodes += 1

        state_id = self._id(states[-1])
        self.id_to_node[state_id] = node
        self.state_tree.insert(state_id, states[-1])

        # when you add a new node also remember
        # to compute their reachable points and put them into the tree
        x_r, u_r = self.model.get_reachable_sampled(states[-1],self.tau)

        for i in range(len(u_r)):
            # x_r represents the whole trajectory to get to the reachable state
            x_r_i = x_r[i][-1]
            x_r_i_id = self._id(x_r_i)

            self.reachable_tree.insert(x_r_i_id, x_r_i)
            self.r_id_to_node[x_r_i_id] = (node, x_r[i], u_r[i])

        return node

    # find x_near
    def nearest_neighbor(self, x_rand):
        # get the nearest reachable point
        id_near, r_near = self.reachable_tree.nearest(x_rand)

        node_near, states, controls = self.r_id_to_node[id_near]
        x_near = node_near.state

        # check if the expansion will be in the direction of x_rand
        # if not, discard
        if np.linalg.norm(x_rand-x_near) < np.linalg.norm(x_rand-r_near):
            return None, None, None
        

        return node_near, states, controls
    
    def expand(self, node_near: Node, states: np.ndarray, controls: np.ndarray):

        x_next: np.ndarray = states[-1]

        _, closest_state = self.state_tree.nearest(x_next)

        if np.linalg.norm(x_next - closest_state) < self.thr:
            # there is already a node at this location
            # TODO consider rewiring if the cost is less
            return None
        
        cost = np.sum( np.linalg.norm(controls) )

        # add node to tree
        node_next = self.add_node(states, controls, cost, node_near)

        return node_next

    def plan(self, max_nodes, plt=None):
        # add the first node with the initial state
        initial_state = self.model.initial_state

        self.initial_node = self.add_node(initial_state)

        goal, distance = self.model.goal_check(self.model.initial_state)
        if distance < self.min_distance:
            self.min_distance = distance
        if goal:
            return True, self.initial_node, self.n_nodes
        
        start = time.time()
        while self.n_nodes < max_nodes:
            t = time.time()
            if self.n_nodes%1000 == 0:
                print(f"n_nodes: {self.n_nodes}, d: {self.min_distance}, t: {t-start} sec", end='\r')

            x_rand = self.model.sample()

            node_near, states, controls = self.nearest_neighbor(x_rand)
            if node_near is None:
                continue # we're not getting close to x_rand

            node_next = self.expand(node_near, states, controls)

            if node_next is None:
                continue

            x_next = node_next.state

            if plt != None: # debug
                x_near = node_near.state
                try: x_rand_plot.remove()
                except: pass
                try: x_next_plot.remove()
                except: pass
                try: x_near_plot.remove()
                except: pass
                x_rand_plot = plt.scatter(x_rand[0], x_rand[1], marker="x", color="green")
                x_near_plot = plt.scatter(x_near[0], x_near[1], color="purple")
                x_next_plot = plt.scatter(x_next[0], x_next[1], color="cyan")
                plt.plot([x_near[0], x_next[0]], [x_near[1],x_next[1]], color="blue")
                plt.draw()
                plt.pause(0.01)
                input()

                plt.scatter(x_near[0], x_near[1], color="blue")
                plt.scatter(x_next[0], x_next[1], color="blue")
               

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
    
    def nodes(self):
        return self.id_to_node.values()
