import numpy as np
from matplotlib.patches import Polygon,Rectangle
from scipy.spatial import ConvexHull
from lib.operations import AH_polytope_vertices 


from scipy import sparse
import pypolycontain as pp
import qpsolvers

def normalize(angle):
    return np.arctan2(np.sin(angle), np.cos(angle))

def plot(root_node, plt, th=100):
    state = root_node.state

    # plot the parent
    plt.scatter(state[0], state[1], c="blue")

    for child in root_node.children:
        child_state = child.state

        norm = np.linalg.norm(state-child_state)

        if norm < th:
            plt.plot([state[0], child_state[0]], [state[1], child_state[1]], c="blue")
        plot(child, plt)

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

    solution = qpsolvers.solve_qp(sP,q,G=sG,h=h,A=sA,b=b, solver="osqp")

    delta = solution[:n_dim]
    x = solution[n_dim:]

    distance = np.linalg.norm(delta)

    point = query + delta

    # point_ = AH.T@x + AH.t.reshape(-1)
    # print(point_ - query - delta) not zero 
    return point, distance

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

            x = qpsolvers.solve_qp(P=np.zeros((m_dim,m_dim)), q =  Gd, G=H, h=h, solver="osqp")
            L[d] = np.dot(Gd,x) + g[d] 

            x = qpsolvers.solve_qp(P=np.zeros((m_dim,m_dim)), q = -Gd, G=H, h=h, solver="osqp")
            U[d] = np.dot(Gd,x) + g[d] 
        
        return AABB(L,U)
    
    def plot_AABB(self,plt,col):
        anchor_point = self.l
        width = abs(self.l[0]-self.u[0])
        heigth = abs(self.l[1]-self.u[1])
        plt.gca().add_patch(Rectangle(anchor_point,
                    width,heigth,
                    edgecolor = col,
                    fill=False,lw=1))

def visualize_polytope_convexhull(polytope,state,color='blue',alpha=0.1,N=20,epsilon=0.001,plt=None):
    v,w=AH_polytope_vertices(polytope,N=N,epsilon=epsilon)
    try:
        v=v[ConvexHull(v).vertices,:]
    except:
        v=v[ConvexHull(w).vertices,:]
    # x = v[0:2,:]
    x = np.append(v[0:2,:],[state],axis=0)
    p=Polygon(v,edgecolor = color,facecolor = color,alpha = alpha,lw=1)
    plt.gca().add_patch(p)


# dai a R3T.expand il plt e inserisci la seguente riga in r3t.py -> riga 79
'''
if plt!=None:
    utils.visualize_polytope_convexhull(polytope,x,plt=plt)
'''