import numpy as np
from utils import normalize

class Pendulum:
    
    motion_primitives = {1.0, 0.0, -1.0}
    # motion_primitives = {1.0}

    def __init__(self, m=1, m_l=0, l=1, g=9.81, b=0.2, initial_state = np.array([0,0]), input_limits=None, dt=0.001):
        
        self.dt = dt

        self.m = m
        self.m_l = m_l
        self.l = l
        self.g = g
        self.b = b
        self.initial_state = initial_state
        self.input_limits = input_limits
        
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

        print(f"Expanding: q_near={q_near}, q_rand={q_rand}")
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

