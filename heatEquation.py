import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

np.set_printoptions(linewidth=1000)
class HeatEquation:
    # source is the source/sink of heat (right hand side)
    # init_val is u(x,0)
    # dirichlet_bound is Dirichlet boundary conditions formatted as a dictionary 
    #     with node number as the key and the boundary condition as the value
    # n is number of nodes
    # l is length of bar
    def __init__(self, source, init_val, dirichlet_bound, n):
        self.source = source
        self.init_val = init_val
        self.n = n
        self.M = np.zeros((n,n))
        self.K = np.zeros((n,n))
        self.h = 1/(n-1)
        self.f = np.zeros((n,1))
        for k in range(n-1):
            m_local = np.zeros((4,4))
            k_local = np.zeros((4,4))
            # Calculate element matrices
            for l in range(2):
                for m in range(2):
                    # Integrate using Simpson's method
                    p1 = self.phi(l, -0.774597) * self.phi(m, -0.774597)
                    p2 = self.phi(l, 0) * self.phi(m, 0)
                    p3 = self.phi(l, 0.774597) * self.phi(m, 0.774597)
                    m_local[l, m] = self.g_quadrature(p1, p2, p3) * 0.5*self.h

                    if l == m:
                        k_local[l, m] = 1/(self.h)
                    else:
                        k_local[l, m] = -1/(self.h)
            # Global assembly
            for l in range(2):
                g_node_1 = self.local2global(k,l)
                for m in range(2):
                    g_node_2 = self.local2global(k,m)
                    self.M[g_node_1, g_node_2] += m_local[l, m]
                    self.K[g_node_1, g_node_2] += k_local[l, m]
        
        self.init_f(0)
        
        # Initial conditions
        self.u = np.zeros((n,1))
        for i in range(n):
            self.u[i,0] = init_val(i*self.h)

        self.dirichlet = dirichlet_bound
        for node, bc in self.dirichlet.items():
            self.u[node] = bc
            self.M[node, :] = 0
            self.M[:, node] = 0
            self.K[node, :] = 0
            self.K[:, node] = 0
            self.M[node, node] = 1
            self.K[node, node] = 1
        
        # print(self.K)
        # print(self.M)
        # print(self.f)
    
    def local2global(self, element, node):
        return element + node

    def phi(self, i, xi):
        if i == 0:
            return (1-xi)*0.5
        return (1+xi)*0.5

    def xi(self, x, elem_num):
        return ((2/self.h)*(x-(elem_num*self.h))-1)

    def x(self, xi, elem_num):
        return ((xi+1)*0.5*self.h + self.h*elem_num)
    
    def simpson(self, p1, p2, p3):
        return 0.5*self.h*(1/3*p1 + 4/3*p2 + 1/3*p3)

    def g_quadrature(self, p1, p2, p3):
        return (5/9)*p1 + (8/9)*p2 + (5/9)*p3

    def init_f(self, t):
        self.f = np.zeros((self.n, 1))
        for k in range(self.n):
            if k < self.n-1:
                p1 = self.source(self.x(-0.774597, k), t)*self.phi(0, -0.774597)*(self.h*0.5)
                p2 = self.source(self.x(0, k), t)*self.phi(0, 0)*(self.h*0.5)
                p3 = self.source(self.x(0.774597, k), t)*self.phi(0, 0.774597)*(self.h*0.5)
                self.f[k,0] += self.g_quadrature(p1, p2, p3)
            if k > 0:
                p1 = self.source(self.x(-0.774597, k-1), t)*self.phi(1, -0.774597)*(self.h*0.5)
                p2 = self.source(self.x(0, k-1), t)*self.phi(1, 0)*(self.h*0.5)
                p3 = self.source(self.x(0.774597, k-1), t)*self.phi(1, 0.774597)*(self.h*0.5)
                self.f[k,0] += self.g_quadrature(p1, p2, p3)

    def iterate(self, dt, end_time, method='f_euler'):
        cur_time = 0
        if method=='f_euler':
            M_inv = np.linalg.inv(self.M)
            coeff_1 = np.identity(self.n) - dt*np.matmul(M_inv,self.K)
            coeff_2 = dt*M_inv
        else:
            B = 1/(dt)*self.M+self.K
            B_inv = np.linalg.inv(B)
            coeff_1 = 1/(dt)*np.matmul(B_inv, self.M)
            coeff_2 = B_inv
            self.init_f(cur_time+dt)
        self.result = self.u.transpose()
        self.t = np.zeros((1,1))
        while cur_time < end_time:
            cur_time += dt
            self.u = np.matmul(coeff_1,self.u) + np.matmul(coeff_2,self.f)
            for node, bc in self.dirichlet.items():
                self.u[node] = bc
            if method =='f_euler':
                self.init_f(cur_time)
            else:
                self.init_f(cur_time+dt)
            self.result = np.append(self.result, self.u.transpose(),axis=0)
            self.t = np.append(self.t, cur_time)

    def plot3D(self, filename=None):
        plt.figure(1)
        node_points = np.linspace(0, 1, num=self.n)
        ax = plt.axes(projection='3d')
        T, X = np.meshgrid(self.t, node_points)
        ax.plot_surface(T, X, self.result.transpose(), cmap=cm.coolwarm)
        ax.set_xlabel('t')
        ax.set_ylabel('x')
        ax.set_zlabel('u')
        ax.set_title('Numerical Solution')
        if filename:
            print("Saving file {}".format(filename))
            plt.savefig(filename)

    def plot_final(self, filename=None):
        plt.figure(2)
        node_points = np.linspace(0, 1, num=self.n)
        ax = plt.axes()
        ax.plot(node_points, self.u)
        ax.set_xlabel('x')
        ax.set_ylabel('u')
        ax.set_title('Final solution at t={}'.format(self.t[-1]))
        if filename:
            print("Saving file {}".format(filename))
            plt.savefig(filename)


def f(x, t):
    return (np.pi**2 -1)*np.exp(-1*t)*np.sin(np.pi*x)

def init_conditions(x):
    return np.sin(np.pi*x)

def analytical_soln(X, T):
    return np.exp(-1*T)*np.sin(np.pi*X)

if __name__ == '__main__':
    n = int(input("Enter number of nodes [Press enter for default:11]:") or 11)
    dt = float(input("Enter time step in decimal form[Press enter for default:1/551]:") or 1/551)
    dirichlet = {0:0, n-1:0}
    heat_eq = HeatEquation(f,init_conditions,dirichlet,n)
    # Forward Euler
    heat_eq.iterate(dt, 1, method='f_euler')
    # Backward Euler
    # heat_eq.iterate(dt, 1, method='b_euler')
    heat_eq.plot_final('final_2d.png')
    heat_eq.plot3D("numerical_soln_3d.png")
    plt.show()