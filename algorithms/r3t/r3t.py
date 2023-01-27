from algorithms.rrt.rrt import StateTree, Node
from rtree import index
import pypolycontain as pp
import utils
import numpy as np

class R3T:
    
    def __init__(self, initial_state, goal_state, eps, state_bounds, solve_input_func, get_polytope_func, get_kpoints_func):
        self.dim = initial_state.shape[0]
        self.initial_state = initial_state
        self.goal_state = goal_state
        self.eps = eps
        self.state_bounds = state_bounds # shape(dim,2)

        self.initial_node = Node(initial_state)

        self.polytope_tree = PolytopeTree(self.dim)
        

        self.solve_input_func = solve_input_func
        self.get_polytope_func = get_polytope_func
        self.get_kpoints_func = get_kpoints_func

        # map
        self.polytope_id_to_node = {}
        self.state_to_node = {}

        self.initial_state_id = hash(str(self.initial_state))

        # insert polytope
        polytope = self.get_polytope_func(initial_state)
        kpoints = self.get_kpoints_func(initial_state)
        self.polytope_tree.insert(polytope, kpoints)
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
    
    def expand(self, node_near: Node, point_near):
        
        x = node_near.state

        x_next, controls = self.solve_input_func(x, point_near)
        
        # add node to tree
        cost = sum([np.linalg.norm(u) for u in controls])
        node_next = Node(x_next, controls, node_near, cost=cost)

        new_child = node_near.add_child(node_next)
        if new_child:
            # add state to database
            state_id = hash(str(x_next))
            # insert polytope
            polytope = self.get_polytope_func(x_next)
            kpoints = self.get_kpoints_func(x_next)
            self.polytope_tree.insert(polytope, kpoints)


            # add link to node
            self.state_to_node[state_id] = node_next
            self.polytope_id_to_node[hash(polytope)] = node_next

            return node_next
        else:
            return None
        
    def plan(self, max_nodes, plt=None):

        goal = self.goal_check(self.initial_state, self.goal_state)
        if goal:
            print("goal")
            return True

        for node in range(max_nodes):

            if node%100 == 0:
                print(node)

            q_rand = self.sample_state()

            r_near, node_near = self.nearest_neighbor(q_rand)
            if r_near is None:
                node -= 1 # iteration does not count
                          # because it did not expand tree
                continue

            node_next = self.expand(r_near, node_near)
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
        polytope_id = self.kpoints_tree.nearest(query)
        polytope_star = self.polytope_id_to_polytope[polytope_id]

        point_star, d_star = utils.distance_point_polytope(query, polytope_star)

        if d_star == 0:
            return polytope_star, d_star, point_star

        # box centered in query large 2d_star
        heuristic_box = utils.AABB(-d_star*np.ones(self.dim) + query, d_star*np.ones(self.dim) + query)
        
        intersecting_polytopes_ids = self.aabb_tree.intersection(heuristic_box)

        # intersecting_polytopes_ids.remove(polytope_id)
        # return polytope_star, d_star, point_star
        while intersecting_polytopes_ids != []:
            idx = intersecting_polytopes_ids.pop()

            P_cand = self.polytope_id_to_polytope[idx]

            p, d = utils.distance_point_polytope(query, P_cand)

            if d < d_star:
                d_star = d
                polytope_star = P_cand
                point_star = p

                # redo the candidates
                # heuristic_box = utils.AABB(-d_star*np.ones(self.dim) + query, d_star*np.ones(self.dim) + query)
                
                # intersecting_polytopes_ids = self.aabb_tree.intersection(heuristic_box) 

                # intersecting_polytopes_ids.remove(idx)
            else: continue

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
        self.AABB_to_polytope[polytope_id] = polytope

    def intersection(self, bbox: utils.AABB):
        
        lu = np.concatenate((bbox.l, bbox.u))

        intersections = list(self.AABB_idx.intersection(lu))

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