from algorithms.rrt.rrt import StateTree, Node
import numpy as np

class RGRRT:
    def __init__(self, initial_state, goal_state, eps, state_bounds, get_reachable_func, tau):
        self.dim = initial_state.shape[0]
        self.initial_state = initial_state
        self.goal_state = goal_state
        self.eps = eps
        self.state_bounds = state_bounds # shape(dim,2)

        self.state_tree = StateTree(self.dim)

        # store the reachable states for every node here
        self.reachable_tree = StateTree(self.dim)

        self.get_reachable_func = get_reachable_func
        self.tau = tau
        # maps
        self.state_to_node = {} # from state_id to node
        self.reachable_to_node = {} # form reachable_id to (node, u)

        # initialization for the first state
        self.initial_node = Node(initial_state)
        self.initial_state_id = hash(str(self.initial_state))
        self.state_to_node[self.initial_state_id] = self.initial_node

        self.state_tree.insert(self.initial_state_id, self.initial_state)
        
        initial_reachable, controls = self.get_reachable_func(self.initial_state, self.tau)
        for reachable, control in zip(initial_reachable, controls):
            reachable_id = hash(str(reachable))
            self.reachable_to_node[reachable_id] = (self.initial_node, control)
            self.reachable_tree.insert(reachable_id, reachable)
        ###


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
        id_near, r_near = self.reachable_tree.nearest(q_rand)

        node_near, u = self.reachable_to_node[id_near]

        q_near = node_near.state

        if np.dot(r_near-q_rand,r_near-q_rand) < np.dot(q_near-q_rand,q_near-q_rand):
            # r_near will be a new node with parent node_near
            return r_near, node_near, u
        else:
            # discard the sampled point
            return None, None, None

    
    def expand(self, q_next, node_near, u):

        # add node to tree
        node_next = Node(q_next, u, node_near, cost=np.linalg.norm(u))

        # add child ot parent
        new_child = node_near.add_child(node_next)
        if new_child:
            # add state to database
            state_id = hash(str(q_next))
            self.state_tree.insert(state_id, q_next)

            # update maps
            self.state_to_node[state_id] = node_next

            reachable, controls = self.get_reachable_func(q_next, self.tau)
            for reachable, control in zip(reachable, controls):
                reachable_id = hash(str(reachable))
                self.reachable_to_node[reachable_id] = (node_next, control)
                self.reachable_tree.insert(reachable_id, reachable)

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

            r_near, node_near, u = self.nearest_neighbor(q_rand)
            if r_near is None:
                node -= 1 # iteration does not count
                          # because it did not expand tree
                continue

            node_next = self.expand(r_near, node_near, u)
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
