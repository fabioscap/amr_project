import numpy as np
from utils import normalize, convex_hull_of_point_and_polytope
import pypolycontain as pp
from models.model import Model

class Pendulum(Model):
    
    motion_primitives = {1,0.0,-1}

    def __init__(self, m=1, m_l=0, l=0.5, g=9.81, b=0.1, 
                       initial_state = np.array([0,0]), 
                       goal_states   = [np.array([np.pi, 0.0]), np.array([-np.pi, 0])],
                       input_limits = np.array([-1.0,1.0]).reshape(1,-1),
                       dt=0.01):
        super().__init__(initial_state, input_limits, dt)

        self.x_dim = 2
        self.u_dim = 1

        self.m = m
        self.m_l = m_l
        self.l = l
        self.g = g
        self.b = b
        self.I = m*l**2 + (m_l*l**2)/12

        self.goal_states = goal_states
        
        self.u_bar =  ( self.input_limits[:,0] + self.input_limits[:,1] )/2
        # self.u_diff = ( self.input_limits[1] - self.input_limits[0] )/2


    def f(self, x:np.ndarray, u:np.ndarray):
        dx = np.zeros_like(x)

        t = -(self.m*self.g*self.l*np.sin(x[0]) + self.m_l*self.g*self.l*np.sin(x[0])/2)

        dx[0] = x[1]
        dx[1] = (1/self.I) * (t + u[0] - self.b*x[1])

        return dx

    # return next state (theta, theta_dot) after applying control u for time dt
    def step(self, x:np.ndarray, u:np.ndarray, dt:float):
        # euler: q_k+1 = q_k + f(q_k, u_k)*dt
        return x + self.f(x,u)*dt
    
    def goal_check(self, x: np.ndarray, eps=0.05) -> tuple[bool, float]:
        
        min_dist = np.inf
        goal = False

        for goal_state in self.goal_states:
            dist = np.linalg.norm(x-goal_state)
            if dist<min_dist:
                min_dist = dist

        if min_dist < eps:
            goal = True
        return goal, min_dist


    def linearize_at(self, x: np.ndarray, u: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        A = np.zeros((self.x_dim, self.x_dim))
        B = np.zeros((self.x_dim, self.u_dim))
        c = np.zeros(self.x_dim)

        A[0,0] = 0
        A[0,1] = 1
        A[1,0] = -(1/self.I)*(self.m*self.g*self.l*np.cos(x[0]) + self.m_l*self.g*self.l*np.cos(x[0])/2)
        A[1,1] = -(1/self.I)*self.b

        A = (np.eye(len(x)) + dt*A)
        B *= dt

        B[0,0] = 0
        B[1,0] = (1/self.I)

        
        c = np.ndarray.flatten(self.step(x,u,dt)) - np.ndarray.flatten(A@x) - np.ndarray.flatten(B*u)

        return A, B, c

    def sample(self, **kwargs) -> np.ndarray:
        rnd = (np.random.rand(2) -0.5)*2 # range between -1 and 1

        rnd[0]*= 3*np.pi/2
        rnd[1]*= 10

        return rnd
    
    def expand_toward(self, x_near:np.ndarray, x_rand:np.ndarray, dt:float)->tuple[np.ndarray, np.ndarray]:
        # expand using pseudoinverse on linearized system
        A, B, c = self.linearize_at(x_near, self.u_bar, dt)

        u = np.linalg.pinv(B)@(x_rand - A@x_near - c)


        # the state has to be actually reachable so I step on the real environment with the systems's dt
        iters = int(dt//self.dt)
        states = np.zeros((iters,self.x_dim))
        controls = np.zeros((iters,self.u_dim))
        x = x_near
        for i in range(iters):
            states[i] = x
            controls[i] = u
            x = self.step(x, u, self.dt)
        
        return states, controls

