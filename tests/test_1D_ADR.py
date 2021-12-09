import sys
sys.path.insert(0, '../Utils/')

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from pinn_ADR import PhysicsInformedNN_ADR
from logger import Logger
from lbfgs import Struct
import time

from scipy import sparse
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
from scipy.interpolate import interp1d

# Manually making sure the numpy random seeds are "the same" on all devices
np.random.seed(17)
#tf.random.set_seed(17)
start_time = time.time()
#############################
# Parameters
#############################
tf_epochs = 200 # Number of training epochs for the regression part
nt_epochs = 2000 # Number of training epochs for the physics part
n_batches = 1 # External batches for Lagrange multipliers optimization
N_boundary = 100 # Number of samples per boundary
N_internal = 1000 # Number of internal collocation points

#############################
# Network architecture
#############################
dim = 1
layers = [dim] + 3*[100] + [1]

#################################
# Setting logger and optimizer
#################################
logger = Logger(frequency=20)
def error():
  return tf.reduce_sum((tf.square(pinn_ADR.return_bc_loss()))).numpy()
logger.set_error_fn(error)

#################################
# Setting up tf optimizer
#################################
tf_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

coupled_optimizer = {}
coupled_optimizer['nt_config'] = Struct()
coupled_optimizer['nt_config'].learningRate = 0.5
coupled_optimizer['nt_config'].maxIter = nt_epochs
coupled_optimizer['nt_config'].nCorrection = 50
coupled_optimizer['nt_config'].tolFun = 1.0 * np.finfo(float).eps
coupled_optimizer['batches'] = n_batches

##############################
# Domain
##############################
x_min, x_max = 0., 1.

# Boundaries (we create a few overlapping boundary points)
left   = np.linspace(x_min,x_min,N_boundary)
right  = np.linspace(x_max,x_max,N_boundary)
x_cord_bc = np.array([left, right])

# Boundary conditions
# If Dirichlet: stores the values of the field boundary condition
# If Homogeneous or Robin: stores the values of the normals
u_bc = np.array([left*0.+0., right*0.+0.])

# Internal points
n_points = N_internal
delta_x = x_max/n_points
nd = 4.0
x = np.linspace(x_min + delta_x/nd, x_max - delta_x/nd, n_points)

# Training data
points_dict = {}
points_dict['x_bc'] = x_cord_bc.flatten()
points_dict['x_eq'] = x.flatten()

u_train_bc = u_bc.flatten()

#####################
# Creating PiNNS
#####################

# Creating PiNNs class
pinn_ADR = PhysicsInformedNN_ADR(layers=layers, optimizer=tf_optimizer, logger=logger,
                                 dim = dim, points_dict=points_dict,
                                 u_bc=u_train_bc, bc_type = 'Dirichlet',
                                 kernel_projection='None',
                                 trainable_kernel=False,
                                 weight_projection=True)

#pinn_ADR.gaussian_bound = 10 #512 #2**3
pinn_ADR.gaussian_scale = 1.

# Adding advection
# velocity = np.concatenate([[points_dict['x_eq']*0.pinp+0.1]], 1).T
# pinn_ADR.add_coupled_variable('velocity', velocity)
# pinn_ADR.add_advection_term('velocity')

# Adding diffusion
#diffusivity = 0.1
diffusivity = x * 0
diffusivity[(x > 0.) & (x <= 0.5)] = 5.
diffusivity[(x > 0.5) & (x <= 1.0)] = 5.
diffusivity = tf.convert_to_tensor(diffusivity, dtype=pinn_ADR.dtype)
pinn_ADR.add_diffusion_term(diffusivity)

# Adding Power
power= x * 0
power[(x > 0.) & (x <= 0.5)] = 500.
power[(x > 0.5) & (x <= 1.0)] = 250.
power = tf.convert_to_tensor(power, dtype=pinn_ADR.dtype)
pinn_ADR.add_coupled_variable('power', power)
pinn_ADR.add_external_reaction_term('power', coef=1.0, ID=0)

# Adding volumetric cooling
#self_reaction_coef = x * 0.
#self_reaction_coef[(x > 0.5) & (x < 0.8)] = -10.
#self_reaction_coef = tf.convert_to_tensor(self_reaction_coef, dtype=pinn_ADR.dtype)
#pinn_ADR.add_self_rection_term(self_reaction_coef)
# pinn_ADR.add_coupled_variable('T_external', x_train_int * 0.)
# pinn_ADR.add_external_reaction_term('T_external', coef=self_reaction_coef, ID=1)

# Training
pinn_ADR.fit(tf_epochs, coupled_optimizer)

class HeatCond:
    def __init__(self, cond, qext, dx, udir, n_ref):
        self.cond = cond
        self.qext = qext
        self.dx = dx
        self.udir = udir
        
        self.repeat_properties(n_ref)
        self.init_mesh()
        self.init_mat_ind()
        
        self.build_LHS_matrix()
        self.build_rhs()
  
    def repeat_properties(self, n_ref):
        # Replecate per zone properties to assign to a mesh
        self.cond = np.repeat(self.cond, n_ref) 
        self.qext = np.repeat(self.qext, n_ref)
        self.dx   = np.repeat(self.dx  , n_ref) / float(n_ref)

    def init_mesh(self):
        # cell interfaces
        x = np.zeros(len(self.dx)+1)
        for i in range(len(self.dx)):
            x[i+1] = x[i] + self.dx[i]
        self.x = x
        # cell mid-point
        xm = x[:-1] + self.dx/2.
        # number of cells, pts
        self.n_cells = len(self.dx)
        self.n_pts   = self.n_cells + 1
  
    def init_mat_ind(self):
        # create indices for rows and columns of the tridiagonal stifness matrix
        indr = np.kron(np.arange(0,self.n_pts),np.ones(3))
        self.indr = indr[1:-1]
        
        indc = np.zeros((self.n_pts,3))
        indc[0,:] = np.array([-1,0,1],dtype=int)
        for i in range (1,self.n_pts):
            indc[i,:] = indc[i-1,:]+1
        indc = indc.flatten()
        self.indc = indc[1:-1]

    def build_LHS_matrix(self):
        # matrix
        L = np.zeros(self.n_pts)
        L[1:] = -self.cond/self.dx
        R = np.zeros(self.n_pts)
        R[:-1] = -self.cond/self.dx
        D = np.zeros(self.n_pts)
        D[:-1] -= R[:-1] 
        D[1:]  -= L[1:] 
        arr = np.vstack((L,D,R)).T.flatten()[1:-1]
        self.A = csr_matrix( (arr, (self.indr, self.indc)), shape=(self.n_pts, self.n_pts) )

    def apply_dirchilet(self):
        # apply right bc (Dirichlet)
        self.A[-1,-2:]=0
        self.A[-1,-1]=1
        self.A[0,1]=0
        self.A[0,0]=1

    def build_rhs(self):
        # forward rhs
        self.rhs = np.zeros(self.n_pts)
        self.rhs[:-1] += self.qext*self.dx/2
        self.rhs[1:]  += self.qext*self.dx/2
        # apply bc
        self.rhs[-1]= self.udir
        self.rhs[0]= self.udir

    def solve(self):
        # solve
        self.u  = spsolve(self.A, self.rhs)
        return self.u, self.x

    def evaluate_u(self, point):
        f = interp1d(self.x, self.u)
        return f(point)


Diff = np.array([5,5])
qext = np.array([500,250]) 
dx   = np.array([0.5,0.5])

Cond = HeatCond(Diff,qext,dx,0,100)
Cond.apply_dirchilet()
us, xs = Cond.solve()

######################################
# Plot
######################################
sol = pinn_ADR.predict(np.array([x]).T)

# Plot
fig_1 = plt.figure(1, figsize=(18, 5))
plt.subplot(1, 3, 1)
plt.plot(x, sol)
plt.xlabel('$x$')
plt.ylabel('$T$')
plt.title(r'Predicted')   
plt.tight_layout()

plt.subplot(1, 3, 2)
plt.plot(xs, us)
plt.xlabel('$x$')
plt.ylabel('$T$')
plt.title(r'Exact')
plt.tight_layout()

plt.subplot(1, 3, 3)
plt.plot(x, np.abs(sol - Cond.evaluate_u(x)))
plt.xlabel('$x$')
plt.ylabel('$T$')
plt.title('Absolute error')
plt.tight_layout()
plt.savefig('lbfgs_heterogenous_nn.jpg')

plt.figure()
plt.semilogy(pinn_ADR.logger.loss_hist)
plt.ylabel('Loss')
plt.xlabel('epoch')
plt.tight_layout()
plt.savefig('loss.jpg')