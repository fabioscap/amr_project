import numpy as np
from utils import normalize, AHPolytope, unitbox

class Pendulum:
    
    motion_primitives = {1, 0.75, 0.5, 0.0, -0.5, -0.75,  -1}
    # motion_primitives = {1.0}

    x_dim = 2
    u_dim = 1

    def __init__(self, m=1, m_l=0, l=0.5, g=9.81, b=0.1, 
                 initial_state = np.array([0,0]), 
                 input_limits=np.array([-1,1]), 
                 dt=0.001):
        
        self.dt = dt

        self.m = m
        self.m_l = m_l
        self.l = l
        self.g = g
        self.b = b
        self.initial_state = initial_state
        self.input_limits = input_limits
        
        self.u_bar =  ( self.input_limits[0] + self.input_limits[1] )/2
        self.u_diff = ( self.input_limits[1] - self.input_limits[0] )/2


        self.I = m*l**2 + (m_l*l**2)/12

    def f(self, q, u):
        dq = np.zeros(2)

        t = -(self.m*self.g*self.l*np.sin(q[0]) + self.m_l*self.g*self.l*np.sin(q[0])/2)

        dq[0] = q[1]
        dq[1] = (1/self.I) * (t + u - self.b*q[1])

        return dq

    # return next state (theta, theta_dot) after applying control u for time dt
    def step(self, q, u):
        # euler: q_k+1 = q_k + f(q_k, u_k)*dt

        q_next = q + self.f(q,u)*self.dt
        q_next[0] = normalize(q_next[0])
        return q_next
    
    # return q_new
    def extend_to(self, q_near, q_rand):
        min_d = np.inf
        q_next = None
        u = None

        for control in self.motion_primitives:
            q_cand = self.step(q_near, control)
            delta = q_rand-q_cand
            delta[0] = normalize(delta[0])

            # TODO find better metrics
            d = np.linalg.norm(delta)
            if d <= min_d:
                q_next = q_cand
                u = control
                min_d = d

        return q_next, u
    
    def get_reachable_points(self, state):

        states = []
        controls = []

        for control in self.motion_primitives:
            cand = self.step(state, control)
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

        c = self.f(x,u) - A@x - B*u

        return A, B, c
    
    def get_reachable_AH(self, x):
        A, B, c = self.linearize_at(x, self.u_bar)

        x_next = (A*self.dt + np.eye(A.shape[0]))@x + B*self.dt*self.u_bar + c

        G = B*self.dt*(self.u_diff)

        RDT = AHPolytope(t=x_next.reshape(-1,1),T=G,P=unitbox(N=1))

        # TODO RCT convex hull


