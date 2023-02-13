import numpy as np
from matplotlib.patches import Polygon,Rectangle
from scipy.spatial import ConvexHull
from lib.operations import AH_polytope_vertices 

from scipy import sparse
import pypolycontain as pp
import qpsolvers
from matplotlib import collections as mc


def normalize(angle):
    return np.arctan2(np.sin(angle), np.cos(angle))

def plot_hopper_2d(root_node, plt):
    states = np.array(root_node.states)[:,:2]
    state = root_node.state[:2]
    plt.scatter(states[:-1,0], states[:-1,1],c="green",s=2)
    plt.plot(states[:,0], states[:,1], linewidth=0.5,c="green")
    for child in root_node.children:
    
        child_state = child.states[0][:2]

        plt.plot([state[0], child_state[0]], [state[1], child_state[1]], linewidth=0.5, c="blue")
        plt.scatter(child_state[0], child_state[1],c="blue",s=2)

        plot_hopper_2d(child, plt)


def plot(nodes, ax,int_color='pink', last_color="red", size=5, lw=1, th=100, plot_all=True):
    # speed up plots by plotting all at once

    lines = []
    lines_int = []

    scatters = []
    scatters_int = []

    for node in nodes:
        # always plot the last node
        scatters.append(node.state)

        # if it has a parent, add a line from this node's state 
        # to the parent's last
        if node.parent is not None:
            x_from = node.parent.state
            x_to   = node.states[0] if plot_all else node.state
            lines.append([x_from,x_to])

        # if plot_all, also plot intermediate states and lines in between them
        n_int = len(node.states)
        if plot_all:
            for i in range(n_int-1): # I already plotted the last state in any case
                scatters_int.append(node.states[i])
                lines_int.append([node.states[i], node.states[i+1]])
        
    scatters = np.array(scatters)
    lines = np.array(lines)
    lc = mc.LineCollection(lines, color=last_color,zorder=2,linewidth=lw)

    if plot_all and len(scatters_int) > 0 :
        scatters_int = np.array(scatters_int)
        ax.scatter(scatters_int[:,0], scatters_int[:,1], color=int_color, s=size*3/5, zorder=3)
        lines_int = np.array(lines_int)
        lc_int = mc.LineCollection(lines_int, color=int_color,zorder=1, linewidth=lw)
        ax.add_collection(lc_int)



    ax.add_collection(lc)
    ax.scatter(scatters[:,0], scatters[:,1], color=last_color, s=size, zorder=4)
    
def distance_point_polytope(query:np.ndarray, AH:pp.AH_polytope):
    n_dim = query.reshape(-1).shape[0]

    AH = pp.to_AH_polytope(AH)

    q_dim, m_dim = AH.P.H.shape

    P = np.zeros((n_dim+m_dim,n_dim+m_dim))
    P[:n_dim,:n_dim] = 0.5*np.eye(n_dim)

    q = np.zeros(n_dim+m_dim).reshape(-1)

    G = np.zeros((q_dim,n_dim+m_dim))
    G[:,n_dim:] = AH.P.H

    h = AH.P.h.reshape(-1)

    A = np.zeros((n_dim, n_dim+m_dim)) 
    A[:,:n_dim] = - np.eye(n_dim)
    A[:,n_dim:] = AH.T

    sA = sparse.csr_matrix(A)
    sP = sparse.csr_matrix(P)
    sG = sparse.csr_matrix(G)

    b = (query.reshape(-1) - AH.t.reshape(-1)).reshape(-1)

    solution = qpsolvers.solve_qp(sP,q,G=sG,h=h,A=sA,b=b, solver="gurobi")
    try:
        delta = solution[:n_dim]
        return delta
    except TypeError:
        print(query)
        print(AH.t)
        print(AH.T)
        print("----")
        print(sP)
        print(q)
        print(sG)
        print(h)
        print(sA)
        print(b)
        return None

class AABB: # axis aligned bounding box

    def __init__(self, l, u):
        assert l.shape == u.shape
    
        self.l = l
        self.u = u
    
    @staticmethod
    def from_AH(AH:pp.AH_polytope):
        
        n_dim = AH.n

        H = AH.P.H
        h = AH.P.h

        m_dim = AH.P.H.shape[1]

        G = AH.T
        g = AH.t

        U = np.zeros(n_dim)
        L = np.zeros(n_dim)

        for d in range(n_dim):
            # Gd: d-esima riga di G
            # dot(Gd,x) == qTx

            Gd = G[d,:]

            x = qpsolvers.solve_qp(P=np.zeros((m_dim,m_dim)), q =  Gd, G=H, h=h, solver="gurobi")
            L[d] = np.dot(Gd,x) + g[d] 

            x = qpsolvers.solve_qp(P=np.zeros((m_dim,m_dim)), q = -Gd, G=H, h=h, solver="gurobi")
            U[d] = np.dot(Gd,x) + g[d] 
        
        return AABB(L,U)
    
    def plot_AABB(self,plt,col, plot = None):
        if plot != None:
            plot.remove()
        anchor_point = self.l
        width = abs(self.l[0]-self.u[0])
        heigth = abs(self.l[1]-self.u[1])
        plot = plt.gca().add_patch(Rectangle(anchor_point,
                    width,heigth,
                    edgecolor = col,
                    fill=False,lw=1))
        return plot
        

def visualize_polytope_convexhull(polytope,state,color='blue',alpha=0.4,N=20,epsilon=0.001,plt=None):
    v,w=AH_polytope_vertices(polytope,N=N,epsilon=epsilon)
    try:
        v=v[ConvexHull(v).vertices,:]
    except:
        v=v[ConvexHull(w).vertices,:]
    # x = v[0:2,:]
    x = np.append(v,[state],axis=0)
    p=Polygon(x,edgecolor = color,facecolor = color,alpha = alpha,lw=1)
    plt.gca().add_patch(p)

def convex_hull_of_point_and_polytope(x, Q):
    r"""
    Inputs:
        x: numpy n*1 array
        Q: AH-polytope in R^n
    Returns:
        AH-polytope representing convexhull(x,Q)
    
    .. math::
        \text{conv}(x,Q):=\{y | y= \lambda q + (1-\lambda) x, q \in Q\}.
    """
    Q=pp.to_AH_polytope(Q)
    q=Q.P.H.shape[1]
    new_T=np.hstack((Q.T,Q.t-x))
    new_t=x
    new_H_1=np.hstack((Q.P.H,-Q.P.h))
    new_H_2=np.zeros((2,q+1))
    new_H_2[0,q],new_H_2[1,q]=1,-1
    new_H=np.vstack((new_H_1,new_H_2))
    new_h=np.zeros((Q.P.h.shape[0]+2,1))
    new_h[Q.P.h.shape[0],0],new_h[Q.P.h.shape[0]+1,0]=1,0
    new_P=pp.H_polytope(new_H,new_h)
    return pp.AH_polytope(t=new_t,T=new_T,P=new_P)