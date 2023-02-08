import numpy as np

class Model:

    def __init__(self, initial_state: np.ndarray, # shape (x_dim, )
                       input_limits:  np.ndarray,  # shape (u_dim, 2)
                       dt: float,
                       ):
        self.initial_state = initial_state.reshape(-1)
        self.input_limits = input_limits
        self.dt = dt

    def step(self, x: np.ndarray, u: np.ndarray, dt: float)->np.ndarray:
        raise NotImplementedError()

    def goal_check(self, x:np.ndarray, eps:float)->tuple[bool, float]:
        raise NotImplementedError()
    
    def sample(self, **kwargs)->np.ndarray:
        raise NotImplementedError()

    def expand_toward(self, x_near:np.ndarray, x_rand:np.ndarray, dt:float)->tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError()
    
    def linearize_at(self, x:np.ndarray, u:np.ndarray, dt:float)->tuple[np.ndarray,  # A, shape (x_dim, x_dim)
                                                                        np.ndarray,  # B, shape (x_dim, u_dim)
                                                                        np.ndarray]: # c, shape (x_dim, )
        raise NotImplementedError()
    
    def get_reachable_sampled(self, x:np.ndarray, dt:float)->tuple[np.ndarray, np.ndarray]:
        # returns a set of sampled points associated with the respective inputs
        # these points are generatet by stepping in the dynamics starting from x and applying
        # a predefined discrete set of inputs
        raise NotImplementedError()