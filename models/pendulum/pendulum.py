import numpy as np
from utils import normalize, convex_hull_of_point_and_polytope
import pypolycontain as pp

class Pendulum:
    
    motion_primitives = {1,0.0,-1}
    # motion_primitives = {1.0}

    x_dim = 2
    u_dim = 1

    def __init__(self, m=1, m_l=0, l=0.5, g=9.81, b=0.1, 
                 initial_state = np.array([0,0]), 
                 dt=0.001):
        
        self.dt = dt

        self.m = m
        self.m_l = m_l
        self.l = l
        self.g = g
        self.b = b
        self.initial_state = initial_state
        self.input_limits = np.array([min(self.motion_primitives), max(self.motion_primitives)])
        
        self.u_bar =  ( self.input_limits[0] + self.input_limits[1] )/2
        self.u_diff = ( self.input_limits[1] - self.input_limits[0] )/2

        self.I = m*l**2 + (m_l*l**2)/12

    def f(self, q, u):
        dq = np.zeros(2)

        t = -(self.m*self.g*self.l*np.sin(q[0]) + self.m_l*self.g*self.l*np.sin(q[0])/2)

        dq[0] = q[1]
        dq[1] = (1/self.I) * (t + u - self.b*q[1])

        return dq.reshape(-1)

    # return next state (theta, theta_dot) after applying control u for time dt
    def step(self, q, u):
        q = q.flatten()
        # euler: q_k+1 = q_k + f(q_k, u_k)*dt
        q_next = q + self.f(q,u)*self.dt
        q_next[0] = normalize(q_next[0])
        return q_next.reshape(-1)
    
    def calc_input(self, x_start, x_c, tau):
        
        iters = int(tau // self.dt)

        A, B, c = self.linearize_at(x_start, self.u_bar)

        A*= tau
        B*= tau
        c*= tau
        u = np.linalg.pinv(B)@(x_c.flatten() - x_start.flatten() - (A@x_start).flatten() - c.flatten())

        # due to linearization u might break the limits
        # so we clamp it
        # u = min(self.input_limits[1], max(self.input_limits[0], u))
        if u < self.input_limits[0]:
            u = self.input_limits[0]
        elif u > self.input_limits[1]:
            u = self.input_limits[1]
        
        x = x_start
        controls = []
        states = [x]
        for _ in range(iters):
            
            x = self.step(x, u).reshape(-1)
            controls.append(u)
            states.append(x)
        
        return states, controls



    # return q_new
    def extend_to(self, q_near, q_rand, tau):
        min_d = np.inf
        q_next = None
        u = None

        for control in self.motion_primitives:
            T = 0
            q_cand = q_near.copy()
            while T<tau:
                q_cand = self.step(q_cand, control)
                T += self.dt
            delta = q_rand-q_cand
            delta[0] = normalize(delta[0])

            # TODO find better metrics
            d = np.linalg.norm(delta)
            if d <= min_d:
                q_next = q_cand
                u = control
                min_d = d
        return q_next.reshape(-1), u
    
    def get_reachable_points(self, state, tau):

        states = []
        controls = []

        for control in self.motion_primitives:
            T = 0
            cand = state
            while T<tau:
                cand = self.step(cand, control)
                T += self.dt
            states.append(cand)
            controls.append(control)

        return states, controls
        
    def linearize_at(self, x, u):
        A = np.zeros((self.x_dim, self.x_dim))
        B = np.zeros((self.x_dim, self.u_dim))
        c = np.zeros((self.x_dim))

        A[0,0] = 0
        A[0,1] = 1
        A[1,0] = -(1/self.I)*(self.m*self.g*self.l*np.cos(x[0]) + self.m_l*self.g*self.l*np.cos(x[0])/2)
        A[1,1] = -(1/self.I)*self.b

        B[0,0] = 0
        B[1,0] = (1/self.I)
        c = np.ndarray.flatten(self.f(x,u)) - np.ndarray.flatten(A@x) - np.ndarray.flatten(B*u)

        return A, B, c
    
    def get_reachable_AH(self, state, tau, convex_hull=True):
        A, B, c = self.linearize_at(state, self.u_bar)
        A = A*tau + np.eye(A.shape[0])
        B *= tau
        c *= tau
        x = np.ndarray.flatten(A@state) + np.ndarray.flatten(B*self.u_bar) + c

        assert(len(x)==len(state))
        G = np.atleast_2d(B*self.u_diff)

        AH = pp.to_AH_polytope(pp.zonotope(G,x))
        if convex_hull:
            state = state.reshape(-1,1) # shape (n,1)
            AH = convex_hull_of_point_and_polytope(state, AH)

        return [], AH

