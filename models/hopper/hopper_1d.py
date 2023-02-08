import numpy as np
import pypolycontain as pp
from utils import convex_hull_of_point_and_polytope

class Hopper1D:

    FLIGHT = 0
    CONTACT = 1
    BOUNCE = -1

    modes = {FLIGHT, CONTACT, BOUNCE}

    def __init__(self, m=1, l=1, p=0.1, b=0.9, g=9.8, f_max=200., epsilon = 1e-7, dt=0.001):
        self.m = m
        self.l = l
        self.p = p
        self.b = b
        self.g = g
        self.f_max = f_max
        self.epsilon = epsilon
        
        self.dt = dt

        self.u_dim = 1

        self.input_limits = np.array([0,f_max])

        self.motion_primitives = {0, f_max/2, f_max}

        # parameter sanity checks
        assert(self.m>0)
        assert(self.l>0)
        assert(self.p>0)
        assert(self.g>0)
        
        assert(self.f_max >= 0)

        self.u_bar = (self.input_limits[1] + self.input_limits[0]) / 2.
        self.u_diff = (self.input_limits[1] - self.input_limits[0]) / 2.

    def f_flight(self, x):
        # not actuated
        dx = np.array([x[1], -self.g])
        return dx

    def f_contact(self, x, u):
        if isinstance(u, np.ndarray):
            u = u[0]
        dx = np.asarray([x[1], u/self.m-self.g])
        return dx
    
    def f_bounce(self, x):
        x_plus = np.array([self.l+self.epsilon, -x[1]*self.b])
        return x_plus


    def check_flight(self, x): return x[0]>self.l+self.p
    def check_contact(self, x): return self.l<x[0] and x[0]<=self.l+self.p
    def check_bounce(self, x): return x[0]<=self.l

    def get_mode(self, x):
        modes = []
        if self.check_flight(x):
            modes.append(self.FLIGHT)
        if self.check_contact(x):
            modes.append(self.CONTACT)
        if self.check_bounce(x):
            modes.append(self.BOUNCE)

        return modes
       

    # return next state (x, x_dot) after applying control u for time dt
    def step(self, x, u, mode=None, dt = None):
        if mode == None:
            mode = self.get_mode(x)[0]
        if dt == None:
            dt = self.dt
        if   mode == Hopper1D.FLIGHT:
            # print("flight")
            x_next = x + self.f_flight(x)*dt
        elif mode == Hopper1D.CONTACT:
            x_next = x + self.f_contact(x,u)*dt
        elif mode == Hopper1D.BOUNCE:
            x_next = self.f_bounce(x)
        else:
            raise Exception()
        return x_next

    def calc_input(self, x_start, x_c, tau):


        A, B, c = self.linearize_at(x_start, self.u_bar, self.get_mode(x_start)[0], tau)

        u =  np.linalg.pinv(B)@(x_c - A@x_start - c)

        # due to linearization u might break the limits
        # so we clamp it
        # u = min(self.input_limits[1], max(self.input_limits[0], u))

        if u < self.input_limits[0]:
            u = self.input_limits[0]
        elif u > self.input_limits[1]:
            u = self.input_limits[1]
        
        x = x_start
        controls = []
        states = []
        iters = int(tau // self.dt)
        for _ in range(iters):
            # print(f"{x} {u}")
            x = self.step(x, u)
            controls.append(u)
            states.append(x)
        # print(f"start {x_start}, goal {x_c} reached {x}")
        return states, controls

    # return q_new
    def extend_to(self, q_near, q_rand, tau):
        raise NotImplementedError
    
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

        
    def linearize_at(self, x, u, mode, dt):
        # the dynamics are already linear
        if mode == self.FLIGHT:
            A = (np.array([[0,1],[0,0]])*dt + np.eye(2))
            B = np.zeros((2,1))*dt
            c = self.step(x,u,Hopper1D.FLIGHT, dt).reshape(-1) - (A@x).reshape(-1) - (B*u).reshape(-1)
        elif mode == self.CONTACT:
            A = (np.array([[0,1],[0,0]])*dt + np.eye(2))
            B = (np.array([[0, 1/self.m]])*dt).reshape(2,1)
            c = np.array([0, -self.g])*dt

        elif mode == self.BOUNCE:
            # xdot_k+1 = -b xdot_k
            B = np.array([[0,0]]).reshape(2,1)
            A = np.array([[0, 0],[0,-self.b]])
            c_ = np.array([self.l+self.epsilon, 0.0])
            c = self.step(x,u,Hopper1D.BOUNCE, dt).reshape(-1) - (A@x).reshape(-1) - (B*u).reshape(-1)
        return A,B,c

    
    def get_reachable_AH(self, state, tau, convex_hull=True):
        available_modes = self.get_mode(state)

        A,B,c = self.linearize_at(state, self.u_bar, available_modes[0], tau)

        states = [state]
        while np.all(B == 0.):

            state = self.step(state, None)
            states.append(state)
            available_modes = self.get_mode(state)
            A,B,c = self.linearize_at(state, self.u_bar, available_modes[0], tau)
        
        # assert np.any(B != 0)
        

        polytopes_list = []
        for mode in self.modes:
            
            if mode not in available_modes:
                continue
   
            A,B,c = self.linearize_at(state, self.u_bar, mode, tau)   

            x = np.ndarray.flatten(A@state) + np.ndarray.flatten(B*self.u_bar) + c

            assert(len(x)==len(state))
            G = np.atleast_2d(B*self.u_diff)

            AH = pp.to_AH_polytope(pp.zonotope(G,x))
            if convex_hull:
                state = state.reshape(-1,1) # shape (n,1)
                AH = convex_hull_of_point_and_polytope(state, AH)

            polytopes_list.append(AH)

        if len(polytopes_list) == 1:
            return states, polytopes_list[0]
        else:
            print(polytopes_list)
            print(self.get_mode(state))
            raise NotImplementedError()
            return np.asarray(polytopes_list)

