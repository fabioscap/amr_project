import numpy as np
import sympy

import pypolycontain as pp
from utils import convex_hull_of_point_and_polytope

class Hopper2D:

    FLIGHT_ASCEND = 0
    FLIGHT_DESCEND = 1
    CONTACT_ASCEND = 2
    CONTACT_DESCEND = 3

    modes = {FLIGHT_ASCEND, FLIGHT_DESCEND, CONTACT_ASCEND, CONTACT_DESCEND}

    def __init__(self, m=5, J=500, m_l=1, J_l=0.5, l1=0.0, l2=0.0, k_g=2e3, b_g=20, g=9.8, dt=0.001):
        '''
        2D hopper with actuated piston at the end of the leg.
        The model of the hopper follows the one described in "Hopping in Legged Systems" (Raibert, 1984)
        '''
        self.dt = dt
        self.m = m
        self.J = J
        self.m_l = m_l
        self.J_l = J_l
        self.l1 = l1
        self.l2 = l2
        self.k_g_y = k_g
        self.k_g_x = 2e3
        self.b_g_x = 200
        self.b_g = b_g
        self.g = g
        self.r0 = 1.5

        # Symbolic variables
        # State variables are s = [x_ft, y_ft, theta, phi, r]
        # x = [s, sdot, x_td]
        # Inputs are u = [tau, chi]

        self.k0 = 800
        self.b_leg = 2
        self.k0_stabilize = 40
        self.b0_stabilize = 10
        self.k0_restore = 60
        self.b0_restore = 15

        self.b_r_ascend = 0.
        self.tau_p = 400.
        self.tau_d = 10.
        
        self.x_ = sympy.MatrixSymbol('x',11,1)
        self.u_ = sympy.MatrixSymbol('u',2, 1)

        self.flight_ascend_dynamics =   self._flight_ascend_dynamics(self.x_)
        self.flight_descend_dynamics =  self._flight_descend_dynamics(self.x_)
        self.contact_ascend_dynamics =  self._contact_ascend_dynamics(self.x_,  self.u_)
        self.contact_descend_dynamics = self._contact_descend_dynamics(self.x_, self.u_)

        self.flight_ascend_J = {
            "A": self.flight_ascend_dynamics.jacobian(self.x_),
            "B": self.flight_ascend_dynamics.jacobian(self.u_)
        }
        self.flight_descend_J = {
            "A": self.flight_descend_dynamics.jacobian(self.x_),
            "B": self.flight_descend_dynamics.jacobian(self.u_)
        }
        self.contact_ascend_J = {
            "A": self.contact_ascend_dynamics.jacobian(self.x_),
            "B": self.contact_ascend_dynamics.jacobian(self.u_)
        }
        self.contact_descend_J = {
            "A": self.contact_descend_dynamics.jacobian(self.x_),
            "B": self.contact_descend_dynamics.jacobian(self.u_)
        }
         
        self.input_limits = np.vstack([[-500,1.4e3], [500,2.5e3]])
        self.u_bar =  ( self.input_limits[0,:] + self.input_limits[1,:] )/2
        self.u_diff = ( self.input_limits[1,:] - self.input_limits[0,:] )/2
        self.motion_primitives = self._build_primitives(n0=3, n1=3)

    def get_ddots(self, x, Fx, Fy, F_leg, u0):
        R = x[4]-self.l1
        # EOM is obtained from Russ Tedrake's Thesis
        a1 = -self.m_l*R
        a2 = (self.J_l-self.m_l*R*self.l1)*sympy.cos(x[2])
        b1 = self.m_l*R
        b2 = (self.J_l -self.m_l*R*self.l1)*sympy.sin(x[2])
        c1 = self.m*R
        c2 = (self.J_l+self.m*R*x[4])*sympy.cos(x[2])
        c3 = self.m*R*self.l2*sympy.cos(x[3])
        c4 = self.m*R*sympy.sin(x[2])
        d1 = -self.m*R
        d2 = (self.J_l+self.m*R*x[4])*sympy.sin(x[2])
        d3 = self.m*R*self.l2*sympy.sin(x[3])
        d4 = -self.m*R*sympy.cos(x[2])
        e1 = self.J_l*self.l2*sympy.cos(x[2]-x[3])
        e2 = -self.J*R
        alpha = (self.l1*Fy*sympy.sin(x[2])-self.l1*Fx*sympy.cos(x[2])-u0)
        A = sympy.cos(x[2])*alpha-R*(Fx-F_leg*sympy.sin(x[2])-self.m_l*self.l1*x[7]**2*sympy.sin(x[2]))
        B = sympy.sin(x[2])*alpha+R*(self.m_l*self.l1*x[7]**2*sympy.cos(x[2])+Fy-F_leg*sympy.cos(x[2])-self.m_l*self.g)
        C = sympy.cos(x[2])*alpha+R*F_leg*sympy.sin(x[2])+self.m*R*(x[4]*x[7]**2*sympy.sin(x[2])+self.l2*x[8]**2*sympy.sin(x[3])-2*x[9]*x[7]*sympy.cos(x[2]))
        D = sympy.sin(x[2])*alpha-R*(F_leg*sympy.cos(x[2])-self.m*self.g)-self.m*R*(2*x[9]*x[7]*sympy.sin(x[2])+x[4]*x[7]**2*sympy.cos(x[2])+self.l2*x[8]**2*sympy.cos(x[3]))
        E = self.l2*sympy.cos(x[2]-x[3])*alpha-R*(self.l2*F_leg*sympy.sin(x[3]-x[2])+u0)

        return sympy.Matrix([(A*b1*c2*d4*e2 - A*b1*c3*d4*e1 - A*b1*c4*d2*e2 + A*b1*c4*d3*e1 + A*b2*c4*d1*e2 - B*a2*c4*d1*e2 - C*a2*b1*d4*e2 + D*a2*b1*c4*e2 + E*a2*b1*c3*d4 - E*a2*b1*c4*d3)/(a1*b1*c2*d4*e2 - a1*b1*c3*d4*e1 - a1*b1*c4*d2*e2 + a1*b1*c4*d3*e1 + a1*b2*c4*d1*e2 - a2*b1*c1*d4*e2), \
                             (A*b2*c1*d4*e2 + B*a1*c2*d4*e2 - B*a1*c3*d4*e1 - B*a1*c4*d2*e2 + B*a1*c4*d3*e1 - B*a2*c1*d4*e2 - C*a1*b2*d4*e2 + D*a1*b2*c4*e2 + E*a1*b2*c3*d4 - E*a1*b2*c4*d3)/(a1*b1*c2*d4*e2 - a1*b1*c3*d4*e1 - a1*b1*c4*d2*e2 + a1*b1*c4*d3*e1 + a1*b2*c4*d1*e2 - a2*b1*c1*d4*e2), \
                            -(A*b1*c1*d4*e2 - B*a1*c4*d1*e2 - C*a1*b1*d4*e2 + D*a1*b1*c4*e2 + E*a1*b1*c3*d4 - E*a1*b1*c4*d3)/(a1*b1*c2*d4*e2 - a1*b1*c3*d4*e1 - a1*b1*c4*d2*e2 + a1*b1*c4*d3*e1 + a1*b2*c4*d1*e2 - a2*b1*c1*d4*e2), \
                             (A*b1*c1*d4*e1 - B*a1*c4*d1*e1 - C*a1*b1*d4*e1 + D*a1*b1*c4*e1 + E*a1*b1*c2*d4 - E*a1*b1*c4*d2 + E*a1*b2*c4*d1 - E*a2*b1*c1*d4)/(a1*b1*c2*d4*e2 - a1*b1*c3*d4*e1 - a1*b1*c4*d2*e2 + a1*b1*c4*d3*e1 + a1*b2*c4*d1*e2 - a2*b1*c1*d4*e2), \
                             (A*b1*c1*d2*e2 - A*b1*c1*d3*e1 - A*b2*c1*d1*e2 - B*a1*c2*d1*e2 + B*a1*c3*d1*e1 + B*a2*c1*d1*e2 - C*a1*b1*d2*e2 + C*a1*b1*d3*e1 + C*a1*b2*d1*e2 + D*a1*b1*c2*e2 - D*a1*b1*c3*e1 - D*a2*b1*c1*e2 - E*a1*b1*c2*d3 + E*a1*b1*c3*d2 - E*a1*b2*c3*d1 + E*a2*b1*c1*d3)/(a1*b1*c2*d4*e2 - a1*b1*c3*d4*e1 - a1*b1*c4*d2*e2 + a1*b1*c4*d3*e1 + a1*b2*c4*d1*e2 - a2*b1*c1*d4*e2)])

    def check_flight_ascend(self, x):
        hip_y_dot = x[6]+x[9]*np.cos(x[2])-x[4]*np.sin(x[2])*x[7]
        return x[1] > 0 and hip_y_dot>0
    def check_flight_descend(self, x):
        hip_y_dot = x[6]+x[9]*np.cos(x[2])-x[4]*np.sin(x[2])*x[7]
        return x[1] > 0 and hip_y_dot<=0
    def check_contact_descend(self, x):
        return x[1] <= 0 and x[9] < 0
    def check_contact_ascend(self, x):
        return x[1] <= 0 and x[9] >= 0

    def get_mode(self, x):
        modes = []
        if self.check_flight_ascend(x): modes.append(self.FLIGHT_ASCEND)
        if self.check_flight_descend(x): modes.append(self.FLIGHT_DESCEND)
        if self.check_contact_ascend(x): modes.append(self.CONTACT_ASCEND)
        if self.check_contact_descend(x): modes.append(self.CONTACT_DESCEND)

        return modes
       
    def _flight_ascend_dynamics(self,x):
        r_diff = x[4]-self.r0
        F_leg_flight = -self.k0_restore*r_diff-self.b0_restore*x[9]
        hip_x_dot = x[5]+x[9]*sympy.sin(x[2])+x[4]*sympy.cos(x[2])*x[7]
        hip_y_dot = x[6]+x[9]*sympy.cos(x[2])-x[4]*sympy.sin(x[2])*x[7]
        alpha_des_ascend = 0.6*sympy.atan(hip_x_dot/(-hip_y_dot-1e-6))

        tau_leg_flight_ascend = (self.tau_p*(alpha_des_ascend-x[2])-self.tau_d*x[7])*-1

        dx = sympy.Matrix([*x[5:-1], *self.get_ddots(x,0,0, F_leg_flight, tau_leg_flight_ascend), x[-1]])

        return dx

    def _flight_descend_dynamics(self, x):
        r_diff = x[4]-self.r0
        F_leg_flight = -self.k0_restore*r_diff-self.b0_restore*x[9]
        hip_x_dot = x[5]+x[9]*sympy.sin(x[2])+x[4]*sympy.cos(x[2])*x[7]
        hip_y_dot = x[6]+x[9]*sympy.cos(x[2])-x[4]*sympy.sin(x[2])*x[7]
        alpha_des_descend = 0.6*sympy.atan(hip_x_dot/(hip_y_dot+1e-6)) # point toward landing point
        tau_leg_flight_descend = (self.tau_p*(alpha_des_descend-x[2])-self.tau_d*x[7])*-1
        
        dx = sympy.Matrix([*x[5:-1],*self.get_ddots(x, 0, 0, F_leg_flight, tau_leg_flight_descend),x[-1]])

        return dx

    def _contact_ascend_dynamics(self, x, u):
        r_diff = x[4]-self.r0
        Fx_contact = -self.k_g_x*(x[0]-x[-1])-self.b_g_x*x[5]
        Fy_contact = -self.k_g_y*(x[1])-self.b_g*x[6]*(1-sympy.exp(x[1]*16))
        F_leg_ascend = -u[1]*r_diff - self.b_r_ascend * x[9]
        tau_leg_contact = u[0]

        dx = sympy.Matrix([*x[5:-1],*self.get_ddots(x,Fx_contact, Fy_contact, F_leg_ascend, tau_leg_contact), x[-1]])

        return dx

    def _contact_descend_dynamics(self, x, u):
        r_diff = x[4]-self.r0
        Fx_contact = -self.k_g_x*(x[0]-x[-1])-self.b_g_x*x[5]
        Fy_contact = -self.k_g_y*(x[1])-self.b_g*x[6]*(1-sympy.exp(x[1]*16))
        F_leg_descend = -self.k0*r_diff-self.b_leg*x[9]
        tau_leg_contact = u[0]

        dx = sympy.Matrix([*x[5:-1],*self.get_ddots(x,Fx_contact, Fy_contact, F_leg_descend, tau_leg_contact), x[-1]])

        return dx

    # return next state (x, x_dot) after applying control u for time dt
    def step(self, x, u):
        # check if state goes from flight mode to contact mode
        # in that case also update x_td which is the last state component
        start_modes = self.get_mode(x)
        x_subs = sympy.Matrix([*list(x)])
        if u is not None:
            u_subs = sympy.Matrix([*list(u)])
        else:
            u_subs = sympy.Matrix([0]*2)
        if start_modes[0] == self.FLIGHT_ASCEND:
            dx = np.array(self.flight_ascend_dynamics.subs(self.x_, x_subs)).reshape(-1).astype(np.float32)
        if start_modes[0] == self.FLIGHT_DESCEND:
            dx = np.array(self.flight_descend_dynamics.subs(self.x_, x_subs)).reshape(-1).astype(np.float32)
        if start_modes[0] == self.CONTACT_ASCEND:
            dx = np.array(self.contact_ascend_dynamics.subs(self.x_, x_subs).subs(self.u_, u_subs)).reshape(-1).astype(np.float32)
        if start_modes[0] == self.CONTACT_DESCEND:
            dx = np.array(self.contact_descend_dynamics.subs(self.x_, x_subs).subs(self.u_, u_subs)).reshape(-1).astype(np.float32)

        x_next = x + dx*self.dt

        end_modes = self.get_mode(x_next)

        if self.FLIGHT_DESCEND in start_modes and self.CONTACT_DESCEND in end_modes:
            input()
            x_next[-1] = x_next[0]

        return x_next


    def calc_input(self, x_start, x_c, tau):
        iters = int(tau // self.dt)

        A, B, c = self.linearize_at(x_start, self.u_bar, self.get_mode(x_start)[0], tau)
        u = np.linalg.pinv(B)@(x_c - x_start - A@x_start - c)

        # due to linearization u might break the limits
        # so we clamp it
        # u = min(self.input_limits[1], max(self.input_limits[0], u))
        """
        if u < self.input_limits[0]:
            u = self.input_limits[0]
        elif u > self.input_limits[1]:
            u = self.input_limits[1]
        """
        x = x_start
        controls = []
        for _ in range(iters):
            # print(f"{x} {u}")
            x = self.step(x, u)
            controls.append(u)
        # print(f"start {x_start}, goal {x_c} reached {x}")
        return x, controls

    # return q_new
    def extend_to(self, q_near, q_rand, tau):
        raise NotImplementedError()
    
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
        if mode == None:
            mode = self.get_mode(x)[0]

        x_subs = sympy.Matrix([*list(x)])
        u_subs = sympy.Matrix([*list(u)])
        if mode == self.FLIGHT_ASCEND:
            A = (np.eye(x.shape[0]) + dt*np.array(self.flight_ascend_J["A"].subs(self.x_, x_subs)).astype(np.float32))
            B = dt*np.array(self.flight_ascend_J["B"].subs(self.x_, x_subs)).astype(np.float32)
            f_bar = np.array(self.flight_ascend_dynamics.subs(self.x_, x_subs)).reshape(-1).astype(np.float32)
            c = dt*(f_bar - A@x - B@u)
        elif mode == self.FLIGHT_DESCEND:
            A = (np.eye(x.shape[0]) + dt*np.array(self.flight_descend_J["A"].subs(self.x_, x_subs)).astype(np.float32))
            B = dt*np.array(self.flight_descend_J["B"].subs(self.x_, x_subs)).astype(np.float32)
            f_bar = np.array(self.flight_descend_dynamics.subs(self.x_, x_subs)).reshape(-1).astype(np.float32)
            c = dt*(f_bar - A@x - B@u)
        elif mode == self.CONTACT_ASCEND:
            A = (np.eye(x.shape[0]) + dt*np.array(self.contact_ascend_J["A"].subs(self.x_, x_subs).subs(self.u_, u_subs)).astype(np.float32))
            B = dt*np.array(self.contact_ascend_J["B"].subs(self.x_, x_subs).subs(self.u_, u_subs)).astype(np.float32)
            f_bar = np.array(self.contact_ascend_dynamics.subs(self.x_, x_subs).subs(self.u_, u_subs)).reshape(-1).astype(np.float32)
            c = dt*(f_bar - A@x - B@u)
        elif mode == self.CONTACT_DESCEND:
            A = (np.eye(x.shape[0]) + dt*np.array(self.contact_descend_J["A"].subs(self.x_, x_subs).subs(self.u_, u_subs)).astype(np.float32))
            B = dt*np.array(self.contact_descend_J["B"].subs(self.x_, x_subs).subs(self.u_, u_subs)).astype(np.float32)
            f_bar = np.array(self.contact_descend_dynamics.subs(self.x_, x_subs).subs(self.u_, u_subs)).reshape(-1).astype(np.float32)
            c = dt*(f_bar - A@x - B@u)
        else:
            return None    

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

            x = np.ndarray.flatten(A@state) + np.ndarray.flatten(B@self.u_bar) + c

            assert(len(x)==len(state))
            G = np.atleast_2d(B@np.diag(self.u_diff))

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

    def _build_primitives(self, n0, n1):
        primitives = []

        u0 = np.linspace(self.input_limits[0,0], self.input_limits[1,0],n0)
        u1 = np.linspace(self.input_limits[0,1], self.input_limits[1,1],n1)
        
        for i in u0:
            for j in u1:
                primitives.append(np.array([i,j]))
        return primitives
    
    def goal_check(self,state, target, tol):
        return np.linalg.norm(state[:-1]-target[:-1]) < tol

