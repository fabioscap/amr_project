from rtree import index
import numpy as np

class StateTree:
    def __init__(self, dim ):
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
    def __init__(self, state, u=None, parent=None, cost=0.):
        self.state = state
        self.u = u
        
        self.parent = parent

        self.cost = cost
        self.children = set()

    def add_child(self, child):
        if child in self.children:
            return False
        else:
            self.children.add(child)
            return True

    def __hash__(self) -> int:
        return hash(str(self.state))
    
    def __eq__(self, __o: object) -> bool:
        return self.__hash__() == __o.__hash__()
    
    def __repr__(self) -> str:
        return f"[{self.state}, {self.u}]"

class RRT:
    
    def __init__(self, initial_state, goal_state, eps, state_bounds, extend_func, tau):
        self.dim = initial_state.shape[0]
        self.initial_state = initial_state
        self.goal_state = goal_state
        self.eps = eps
        self.state_bounds = state_bounds # shape(dim,2)

        self.state_tree = StateTree(self.dim)

        self.extend_func = extend_func
        self.tau = tau

        # map
        self.state_to_node = {}

        self.initial_node = Node(initial_state)
        self.initial_state_id = hash(str(self.initial_state))
        self.state_to_node[self.initial_state_id] = self.initial_node

        self.state_tree.insert(self.initial_state_id, self.initial_state)

        self.min_distance = np.inf

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
        id_near, q_near = self.state_tree.nearest(q_rand)
        return q_near
    
    def expand(self, q_near, q_rand):
        q_next, u = self.extend_func(q_near, q_rand, self.tau)

        if hash(str(q_next)) in self.state_to_node.keys():
            return None

        node_near = self.state_to_node[hash(str(q_near))]

        # add node to tree
        node_next = Node(q_next, u, node_near, cost=np.linalg.norm(u))

        new_child = node_near.add_child(node_next)
        if new_child:
            # add state to database
            state_id = hash(str(q_next))
            self.state_tree.insert(state_id, q_next)

            # add link to node
            self.state_to_node[state_id] = node_next

            return node_next
        else:
            return None

    def plan(self, max_nodes, plt=None):

        goal = self.goal_check(self.initial_state, self.goal_state)
        if goal:
            print("goal")
            return True

        
        for node in range(max_nodes):
            if node%1000 == 0:
                print(node)

            q_rand = self.sample_state()
            q_near = self.nearest_neighbor(q_rand)

            node_next = self.expand(q_near, q_rand)

            if node_next is None:
                node -= 1
                continue
            q_next = node_next.state

            if plt != None:
                q_parent = node_next.parent.state
                plt.scatter(q_next[0], q_next[1], c="blue")
                plt.plot([q_parent[0], q_next[0]],[q_parent[1], q_next[1]], c="blue")


            goal = self.goal_check(q_next, self.goal_state)

            if goal:
                print("goal")
                print(self.min_distance)
                return True, node_next, node
        print("no goal")
        print(self.min_distance)
        return False, None, node
    
    def get_plan(self, node: Node, plt=None):
        plan = [node]
        q = node.state
        while node.parent != None:
            plan = [node.parent] + plan

            node = node.parent

            if plt != None:
                q_next = node.state

                plt.plot([q[0],q_next[0]],[q[1],q_next[1]],c="red")
                q = q_next

        return plan

    def goal_check(self, q, q_goal):
        delta = q-q_goal
        norm = np.linalg.norm(delta)

        if norm < self.min_distance:
            self.min_distance = norm
        return norm < self.eps
