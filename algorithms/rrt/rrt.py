from algorithms.planner import Planner, Node
import numpy as np
from models.model import Model
import time

class RRT(Planner):
    
    def __init__(self, model: Model, tau, thr=1e-9):
        super().__init__(model, tau, thr)

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

        # optimization for hybrid systems
        states = np.vstack(( states , self.model.ffw(states[-1]) ))

        if states is None:
            return None # cannot reach that state
        
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
            if self.n_nodes%100 == 0:
                print(f"n_nodes: {self.n_nodes}, d: {self.min_distance}, t: {t-start} sec", end='\r')

            x_rand = self.model.sample()

            node_near = self.nearest_neighbor(x_rand)

            node_next = self.expand(node_near, x_rand)

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
               

            for i in range(node_next.states.shape[0]):
                state = node_next.states[i,:]
                goal, distance = self.model.goal_check(state)
                if distance < self.min_distance:
                    self.min_distance = distance
                if goal:
                    node_next.states = node_next.states[:i,:]
                    plan = self.get_plan(node_next)
                    return True, plan

        return False, None