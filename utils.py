import numpy as np
from algorithms.rrt.rrt import Node

def normalize(angle):
    return np.arctan2(np.sin(angle), np.cos(angle))

class Polytope:
    def __init__(self, H: np.ndarray, h: np.ndarray):
        # H has shape (q, n)
        # h has shape (q)
        self.H = H
        self.h = h

        # this class is constant, save hash
        self.hash_value = None
    
    def __hash__(self):
        if self.hash_value == None:
            self.hash_value = hash(str(np.hstack([self.H, self.h])))        
        return self.hash_value


class AHPolytope:
    def __init__(self, t: np.ndarray, T: np.ndarray, P: Polytope):
        self.t = t # n
        self.T = T # (n,h)
        self.P = P # Polytope in h dimensions

        self.hash_value = None


    def __hash__(self) -> int:
        if self.hash_value == None:
            self.hash_value = hash(self.P) + hash(str(np.hstack([self.T, self.t]))) 
        return self.hash_value


class AABB: # axis aligned bounding box
    pass

def unitbox(N):
    H=np.vstack((np.eye(N),-np.eye(N)))
    h=np.ones((2*N,1))
    return Polytope(H,h)


def plot(root_node: Node, plt, th=4):
    state = root_node.state

    # plot the parent
    plt.scatter(state[0], state[1], c="blue")

    for child in root_node.children:
        child_state = child.state

        norm = np.linalg.norm(state-child_state)

        if norm < th:
            plt.plot([state[0], child_state[0]], [state[1], child_state[1]], c="blue")
        plot(child, plt)

