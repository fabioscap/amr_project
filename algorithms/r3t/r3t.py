from algorithms.rrt.rrt import StateTree, Node
from rtree import index
import pypolycontain as pp
import utils
import numpy as np

class R3T:
    
    def __init__(self, initial_state, goal_state, eps, state_bounds, solve_input_func, get_polytope_func, get_kpoints_func, tau):
        self.dim = initial_state.shape[0]
        self.initial_state = initial_state
        self.goal_state = goal_state
        self.eps = eps
        self.state_bounds = state_bounds # shape(dim,2)

        self.initial_node = Node(initial_state)

        self.polytope_tree = PolytopeTree(self.dim)
        self.state_tree = StateTree(self.dim)

        self.solve_input_func = solve_input_func
        self.get_polytope_func = get_polytope_func
        self.get_kpoints_func = get_kpoints_func
        self.tau = tau


        # map
        self.polytope_id_to_node = {}
        self.state_to_node = {}

        self.initial_state_id = hash(str(self.initial_state))
        
        # insert polytope
        polytope = self.get_polytope_func(initial_state, self.tau)
        kpoints = self.get_kpoints_func(initial_state, self.tau)
        self.polytope_tree.insert(polytope, kpoints)
        self.state_tree.insert(self.initial_state_id, self.initial_state)

        self.state_to_node[self.initial_state_id] = self.initial_node
        self.polytope_id_to_node[hash(polytope)] = self.initial_node
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
        poly_near, d_near, point_near = self.polytope_tree.nearest(q_rand)
        # possibly return the id
        node_near = self.polytope_id_to_node[hash(poly_near)]

        return node_near, point_near
    
    def expand(self, node_near: Node, point_near,plt):
        
        x = node_near.state

        x_next, controls = self.solve_input_func(x, point_near, self.tau)
        state_id = hash(str(x_next))

        _, closest_state = self.state_tree.nearest(x_next)

        if np.linalg.norm(x_next-closest_state) < 1e-3:
            return None

        # add node to tree
        cost = sum([np.linalg.norm(u) for u in controls])
        node_next = Node(x_next, controls, node_near, cost=cost)

        new_child = node_near.add_child(node_next)

        # insert polytope
        polytope = self.get_polytope_func(x_next, self.tau)
        kpoints = self.get_kpoints_func(x_next, self.tau)
        self.polytope_tree.insert(polytope, kpoints)
        self.state_tree.insert(state_id, x_next)
        if plt!=None:
            utils.visualize_polytope_convexhull(polytope,x,plt=plt)

        # add link to node
        self.state_to_node[state_id] = node_next
        self.polytope_id_to_node[hash(polytope)] = node_next

        return node_next
        
    def plan(self, max_nodes, plt=None):

        goal = self.goal_check(self.initial_state, self.goal_state)
        if goal:
            print("goal")
            return True

        n_nodes = 1
        while n_nodes < max_nodes:
            if n_nodes%100 == 0:
                print("nodes", n_nodes)
                _, c = self.state_tree.nearest(self.goal_state)
                print("dist", np.linalg.norm(c-self.goal_state))

            q_rand = self.sample_state()

            r_near, node_near = self.nearest_neighbor(q_rand)
            if r_near is None:
                continue

            node_next = self.expand(r_near, node_near,plt)
            if node_next is None:
                continue
            n_nodes+= 1
            q_next = node_next.state
            if plt != None:
                q_parent = node_next.parent.state
                plt.scatter(q_next[0], q_next[1], c="blue")
                plt.plot([q_parent[0], q_next[0]],[q_parent[1], q_next[1]], c="blue")
                plt.draw()
                plt.pause(0.05)


            goal = self.goal_check(q_next, self.goal_state)

            if goal:
                print("goal")
                print(self.min_distance)
                return True, node_next, n_nodes
        print("no goal")
        print(self.min_distance)
        return False, None, n_nodes
    
    def goal_check(self, q, q_goal):
        delta = q-q_goal

        norm = np.linalg.norm(delta)
        if norm < self.min_distance:
            self.min_distance = norm
        return norm < self.eps
    
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
        


class PolytopeTree:

    def __init__(self, dim) -> None:

        self.dim = dim

        self.kpoints_tree = KPointTree(dim) # or kd tree
        self.aabb_tree = AABBTree(dim)

        self.polytope_id_to_polytope = {}

    def insert(self, polytope: pp.AH_polytope, kpoints):
        polytope_id = hash(polytope)
        kpoints = kpoints[0] # I also return controls
        for i in range(len(kpoints)):
            kpoint = kpoints[i]
            self.kpoints_tree.insert(polytope_id, kpoint)

        # AABB converts to bbox
        self.aabb_tree.insert(polytope_id, polytope)

        self.polytope_id_to_polytope[polytope_id] = polytope

    def nearest(self, query):
        #AABB query

        # polytope relative to closest kpoint
        id_star = self.kpoints_tree.nearest(query)
        polytope_star = self.polytope_id_to_polytope[id_star]
        
        point_star, d_star = utils.distance_point_polytope(query, polytope_star)
        n_calls = 1
        
        if d_star == 1e-9:
            return polytope_star, d_star, point_star

        # box centered in query large 2d_star
        heuristic_box = utils.AABB(-d_star*np.ones(self.dim)*1.001 + query, d_star*np.ones(self.dim)*1.001 + query)
        
        intersecting_polytopes_ids = self.aabb_tree.intersection(heuristic_box)
        # print("intersecting first box:", len(intersecting_polytopes_ids))

        # intersecting_polytopes_ids.remove(polytope_id)
        # return polytope_star, d_star, point_star
        dropped_polytope_ids = {id_star,}
        
        while intersecting_polytopes_ids != set():
            idx = intersecting_polytopes_ids.pop()

            # spare some calls to the solver
            if idx in dropped_polytope_ids or idx == id_star:
                # print("already did")
                continue
            else:
                P_cand = self.polytope_id_to_polytope[idx]
                p, d = utils.distance_point_polytope(query, P_cand)
                n_calls += 1

            if d < d_star:
                d_star = d
                id_star = idx
                polytope_star = P_cand
                point_star = p

                # redo the candidates
                heuristic_box = utils.AABB(-d_star*np.ones(self.dim)*1.001 + query, d_star*np.ones(self.dim)*1.001 + query)

                intersecting_polytopes_ids = self.aabb_tree.intersection(heuristic_box) 
                dropped_polytope_ids.add(idx)
                # print("intersecting next box:", len(intersecting_polytopes_ids - dropped_polytope_ids))
            else:
                dropped_polytope_ids.add(idx)
        # print("n_calls",n_calls)
        # input()
        return polytope_star, d_star, point_star

class AABBTree:

    def __init__(self, dim):

        self.AABB_tree_p = index.Property()
        self.AABB_tree_p.dimension = dim

        self.AABB_idx = index.Index(properties=self.AABB_tree_p,
                                    interleaved=True)
        
        self.AABB_to_polytope = {}

    def insert(self, polytope_id, polytope: pp.AH_polytope):

        bbox = utils.AABB.from_AH(polytope)

        lu = np.concatenate((bbox.l, bbox.u))
        self.AABB_idx.insert(polytope_id, lu)
        if polytope_id in self.AABB_to_polytope:
            print("ASDASDAS")
        self.AABB_to_polytope[polytope_id] = polytope

    def intersection(self, bbox: utils.AABB):
        
        lu = np.concatenate((bbox.l, bbox.u))

        intersections = set(self.AABB_idx.intersection(lu))

        return intersections

    # test
    def insert_bbox(self, bbox_id, bbox):

        lu = np.concatenate((bbox.l, bbox.u))

        self.AABB_idx.insert(bbox_id, lu)

class KPointTree:
    def __init__(self, dim):
        self.dim = dim
        self.kpoint_tree_p = index.Property()
        self.kpoint_tree_p.dimension = dim
        
        self.kpoint_idx = index.Index(properties=self.kpoint_tree_p)

        # map
        self.kpoint_id_to_polytope = {}

    def insert(self, polytope_id, kpoint):
        kpoint_id = hash(str(kpoint))
        self.kpoint_idx.insert(kpoint_id, kpoint)
        self.kpoint_id_to_polytope[kpoint_id] = polytope_id

    def nearest(self, query):
        # https://github.com/wualbert/r3t/blob/master/common/basic_rrt.py
        # they stack the state for some reason

        nearest_id = list(self.kpoint_idx.nearest(query, num_results=1))[0]
        polytope_id = self.kpoint_id_to_polytope[nearest_id]

        return polytope_id