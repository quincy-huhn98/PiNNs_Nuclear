
# coding: utf-8

# # Basic Configuration

# Inserting path with Utils in the cloned repo
import sys, os
cur = os.path.dirname(os.getcwd())
print(cur+'/Utils')
sys.path.insert(0, cur+'/Utils')


import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
#from pinn_NS import PhysicsInformedNN_NS
from pinn_ADR import PhysicsInformedNN_ADR
from logger import Logger
from lbfgs import Struct

# coding: utf-8

# # load python modules

import numpy as np
import matplotlib.pyplot as plt
snorder = 2
mu_q, w_q = np.polynomial.legendre.leggauss(snorder)
w_q /= np.sum(w_q)
n_dir = snorder
n_moment = 1


# # create 1d slab mesh and data

def create_1d_slab_mesh_data(sigt_,sigs_,qext_,width,n_ref):
    sigt = np.repeat(sigt_, n_ref)
    sigs = np.repeat(sigs_, n_ref)
    qext = np.repeat(qext_, n_ref)
    width= np.repeat(width_,n_ref) / float(n_ref)

    # cell interfaces
    x = np.zeros(len(width)+1)
    for i in range(len(width)):
        x[i+1] = x[i] + width[i]
    # cell width
    dx = np.copy(width)
    # cell mid-point
    xm = x[:-1] + dx/2.
    # number of unknowns
    nel = len(xm)
    #
    return sigt, sigs, qext, x, xm, dx


# # routine to compute total source: scattering + external

def compute_src_moments(phi, sigs, qext):
    # only zero-th moment here
    return sigs * phi + qext


# # routine for transport sweeps (spatial solves)

def transport_sweep(a, nel, sigt, q, psi_bc):
    # mem alloc
    phi = np.zeros(nel) 

    # for plotting
    psi_cell = np.zeros((nel,n_dir))

    # loop over all directions
    for dir_ in range(n_dir):
        # select direction and sweep order
        mu = mu_q[dir_]
        if mu>0:
            ibeg = 0
            iend = nel
            incr = +1
        else:
            ibeg = nel-1
            iend = -1
            incr = -1
            
        # short cut for absolute value of mu
        m = np.fabs(mu)
        # bc
        psi_in = psi_bc[dir_]

        # loop over cells
        for iel in range(ibeg, iend, incr):
            m_dx = m/dx[iel] # shortcut
            # compute exiting psi
            psi_out = ( q[iel] + ( m_dx - (1-a)*sigt[iel] ) * psi_in ) / ( m_dx + a*sigt[iel] )
            psi_ave = (1-a)*psi_in + a*psi_out

            # store cell-average psi (only for plotting)
            psi_cell[iel,dir_] = psi_ave

            # accumulate the scalar flux integral
            phi[iel] += w_q[dir_] * psi_ave
                
            # prepare for next cell
            psi_in = psi_out
            
    return phi, psi_cell


# # routine for full transport solve (update total source + perform sweeps + rinse and repeat until converged)

def solve(sigt, sigs, qext, dx, psi_bc, aFD=0.5, SI_tol=1e-4, SI_max=10, verbose=0):
# initialize flux moments (can compute more than needed for scattering src eval)
    nel = len(dx)
    phi = np.zeros(nel)
    
    # compute initial total source
    src_mom = compute_src_moments(phi, sigs, qext)
       
    # source iteration loop
    for SI_iter in range(SI_max):
        
        # save old scalar flux for error convergence
        phi_old = np.copy(phi)
        # perform sweeps
        phi,psi_c = transport_sweep(aFD, nel, sigt, src_mom, psi_bc)
        
        # compute error in successive iterates of the scalar flux
        err_ = np.linalg.norm(phi-phi_old)
        
        # printout
        if verbose>0:
            if verbose>99 or SI_iter%int(pow(10,verbose))==0:
                print("Iteration {0:>5}, error = {1:3.5e}".format(SI_iter,err_))                
        
        # convergence check
        if err_ < SI_tol:
            print("Iteration {0:>5}, error = {1:3.5e}".format(SI_iter,err_))                
            print("Converged")
            break
        else:
            # update total source
            src_mom = compute_src_moments(phi, sigs, qext)
            
        # warn
        if SI_iter == SI_max-1:
            print("Iteration {0:>5}, error = {1:3.5e}".format(SI_iter,err_))                
            print('Not enough SI')
        
    return phi, psi_c

#############################
# Parameters
#############################
tf_epochs = 100 # Number of training epochs for the regression part
nt_epochs = 2000 # Number of training epochs for the physics part
n_batches = 1 # External batches for Lagrange multipliers optimization
N_boundary = 50 # Number of samples per boundary
N_internal = 1000 # Number of internal collocation points

#############################
# Network srchitecture
#############################
dim = 1
layers = [dim] + 5*[64] + [1]

# Setting seeds
np.random.seed(17)
tf.random.set_seed(17)


##############################
# Parameters
##############################
# # Reference
sigt = np.array([5., 5., 5., 5.,  5. ])
sigs = np.array([ 0., 0., 0., 0, 0])
qext = np.array([5., 5., 5., 5. , 5. ])
width= np.array([ 2., 1., 2., 1. , 2. ])

agg_width = np.cumsum(width)

snorder = 2
mu_q, w_q = np.polynomial.legendre.leggauss(snorder)
w_q /= np.sum(w_q)

##############################
# Domain
##############################
x_min, x_max = 0., 5.

# Boundaries (we create a few overlapping boundary points)
left   = np.linspace(x_min,x_min,N_boundary)
right  = np.linspace(x_max,x_max,N_boundary)

# Boundary conditions
# If Dirichlet: stores the values of the field boundary condition
# If Homogeneous or Robin: stores the values of the normals
u_bc_left  = np.array([left*0.])
u_bc_right = np.array([right*0.])

# Internal points
n_points = N_internal
x_mesh = np.linspace(x_min, x_max, n_points)

# Internal points training dict
points_dict = {}
points_dict['x_eq'] = x_mesh.flatten()

#################################
# Setting logger and optimizer
#################################
logger_pinn = {}
for i, _ in enumerate(mu_q):
  logger_pinn[i] = Logger(frequency=20)
  def error():
    return tf.reduce_sum((tf.square(dict_pinns[i].return_bc_loss()))).numpy()
  logger_pinn[i].set_error_fn(error)

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


#####################
# Creating PiNNS
#####################

def predict_PiNN_dir(i, x):
  return dict_pinns[i].predict(np.array([x]).T)

def compute_scalar_flux(dict_pinns):
  scalar_flux = np.zeros_like(x_mesh)
  for i, weight in enumerate(w_q):
    scalar_flux += predict_PiNN_dir(i, x_mesh) * weight
  return scalar_flux


dict_pinns = {}
for i, direction in enumerate(mu_q):

  #Creating PiNNs class
  if direction > 0:
    x_cord_bc = np.array([left])
    points_dict['x_bc'] = x_cord_bc.flatten()
    u_train_bc = u_bc_left.flatten()
  else:
    x_cord_bc = np.array([right])
    points_dict['x_bc'] = x_cord_bc.flatten()
    u_train_bc = u_bc_right.flatten()


  dict_pinns[i] = PhysicsInformedNN_ADR(layers=layers, optimizer=tf_optimizer, logger=logger_pinn[i], 
                                  dim = dim, points_dict=points_dict, 
                                  u_bc=u_train_bc, bc_type = 'Dirichlet',
                                  kernel_projection='Fourier',
                                  trainable_kernel=False)
  dict_pinns[i].gaussian_scale = 50.

  # Adding advection
  velocity = np.concatenate([[points_dict['x_eq']*0. + direction]], 1).T
  dict_pinns[i].add_coupled_variable('velocity', velocity)
  dict_pinns[i].add_advection_term('velocity')

  # Adding Power
  power = x_mesh * 0.
  power[x_mesh < agg_width[0]] = qext[0]
  power[(x_mesh >= agg_width[0]) & (x_mesh < agg_width[1])] = qext[1]
  power[(x_mesh >= agg_width[1]) & (x_mesh < agg_width[2])] = qext[2]
  power[(x_mesh >= agg_width[2]) & (x_mesh < agg_width[3])] = qext[3]
  power[(x_mesh >= agg_width[3])] = qext[4]
  power = tf.convert_to_tensor(power, dtype=dict_pinns[i].dtype)
  dict_pinns[i].add_coupled_variable('power', power)
  dict_pinns[i].add_external_reaction_term('power', coef=1.0, ID=0)

  # Adding self-reaction
  self_reaction_coef = x_mesh * 0. + 1.
  self_reaction_coef[x_mesh < agg_width[0]] = sigt[0]
  self_reaction_coef[(x_mesh >= agg_width[0]) & (x_mesh < agg_width[1])] = sigt[1]
  self_reaction_coef[(x_mesh >= agg_width[1]) & (x_mesh < agg_width[2])] = sigt[2]
  self_reaction_coef[(x_mesh >= agg_width[2]) & (x_mesh < agg_width[3])] = sigt[3]
  self_reaction_coef[(x_mesh >= agg_width[3])] = sigt[4]
  self_reaction_coef = tf.convert_to_tensor(self_reaction_coef, dtype=dict_pinns[i].dtype)
  dict_pinns[i].add_self_rection_term(self_reaction_coef)

for i, _ in enumerate(mu_q):
  dict_pinns[i].fit(tf_epochs, coupled_optimizer, restart_model = True)

scalar_flux = compute_scalar_flux(dict_pinns)

sigt_ = np.array([5.])
sigs_ = np.array([ 0.])
qext_ = np.array([5.])
width_= np.array([ 5])


# ### boundary values

psi_bc = np.zeros(n_dir)


# ## run and plot

SI_max = 10000
SI_tol = 1e-8

n_ref=1000
sigt,sigs,qext,x,xm,dx = create_1d_slab_mesh_data(sigt_,sigs_,qext_,width_,n_ref)

Phi, Psi_cell = solve(sigt, sigs, qext, dx, psi_bc, aFD=0.5, SI_tol=SI_tol, SI_max=SI_max, verbose=1)

# In[22]:


plt.figure()
plt.plot(x_mesh, predict_PiNN_dir(0, x_mesh), label='Pinn Dir. 0')
plt.plot(x_mesh, predict_PiNN_dir(1, x_mesh), label='Pinn Dir. 1')
for dir_ in range(len(mu_q)):
    plt.plot(xm, Psi_cell[:,dir_], '--', label='FD Dir. '+str(dir_))
plt.legend()
plt.grid()
plt.savefig('afPolzup1.jpg')

plt.figure()
plt.plot(x_mesh, scalar_flux, label='Pinn', linewidth=2.)
plt.plot(xm, Phi, '--', label='FD')
plt.legend()
plt.grid()
plt.savefig('sfPolzup1.jpg')


# # ## Reed problem with scattering

# # ### S2

# # In[ ]:


#############################
# Parameters
#############################
tf_epochs = 20 # Number of training epochs for the regression part
nt_epochs = 40 # Number of training epochs for the physics part
n_batches = 1 # External batches for Lagrange multipliers optimization
N_boundary = 50 # Number of samples per boundary
N_internal = 1000 # Number of internal collocation points

#############################
# Network srchitecture
#############################
dim = 1
layers = [dim] + 5*[64] + [1]

# Setting seeds
np.random.seed(17)
tf.random.set_seed(17)


##############################
# Parameters
##############################
# # Reference
sigt = np.array([5.0, 5.0 ])
sigs = np.array([2.5, 2.5 ])
qext = np.array([5.0, 0.0 ])
width= np.array([0.5, 0.5 ])

agg_width = np.cumsum(width)

snorder = 2
mu_q, w_q = np.polynomial.legendre.leggauss(snorder)
w_q /= np.sum(w_q)

##############################
# Domain
##############################
x_min, x_max = 0., 1.

# Boundaries (we create a few overlapping boundary points)
left   = np.linspace(x_min,x_min,N_boundary)
right  = np.linspace(x_max,x_max,N_boundary)

# Boundary conditions
# If Dirichlet: stores the values of the field boundary condition
# If Homogeneous or Robin: stores the values of the normals
u_bc_left  = np.array([left*0.])
u_bc_right = np.array([right*0.])

# Internal points
n_points = N_internal
x_mesh = np.linspace(x_min, x_max, n_points)

# Internal points training dict
points_dict = {}
points_dict['x_eq'] = x_mesh.flatten()

#################################
# Setting logger and optimizer
#################################
logger_pinn = {}
for i, _ in enumerate(mu_q):
  logger_pinn[i] = Logger(frequency=200)
  def error():
    return tf.reduce_sum((tf.square(dict_pinns[i].return_bc_loss()))).numpy()
  logger_pinn[i].set_error_fn(error)

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


#####################
# Creating PiNNS
#####################

def predict_PiNN_dir(i, x):
  return dict_pinns[i].predict(np.array([x]).T)

def compute_scalar_flux(dict_pinns):
  scalar_flux = np.zeros_like(x_mesh)
  for i, weight in enumerate(w_q):
    scalar_flux += predict_PiNN_dir(i, x_mesh) * weight
  return scalar_flux

def compute_scattering_vector():
  agg_width = np.cumsum(width)
  scattering_vector = x_mesh * 0.
  scattering_vector[x_mesh < agg_width[0]] = sigs[0]
  for i in range(1,len(sigs)):
    scattering_vector[(x_mesh >= agg_width[i-1]) & (x_mesh < agg_width[i])] = sigs[i]
  scattering_vector[(x_mesh >= agg_width[-1])] = sigs[-1]
  return scattering_vector

def compute_scattering_source(dict_pinns):
  return compute_scattering_vector() * compute_scalar_flux(dict_pinns) #/ 2.0

def add_scattering_sources(dict_pinns):
  scat_source = compute_scattering_source(dict_pinns)
  for i, _ in enumerate(mu_q):
    dict_pinns[i].add_coupled_variable('scat_source', scat_source)
    dict_pinns[i].add_external_reaction_term('scat_source', coef=1.0, ID=1)


dict_pinns = {}
for i, direction in enumerate(mu_q):

  #Creating PiNNs class
  if direction > 0:
    x_cord_bc = np.array([left])
    points_dict['x_bc'] = x_cord_bc.flatten()
    u_train_bc = u_bc_left.flatten()
  else:
    x_cord_bc = np.array([right])
    points_dict['x_bc'] = x_cord_bc.flatten()
    u_train_bc = u_bc_right.flatten()


  dict_pinns[i] = PhysicsInformedNN_ADR(layers=layers, optimizer=tf_optimizer, logger=logger_pinn[i], 
                                  dim = dim, points_dict=points_dict, 
                                  u_bc=u_train_bc, bc_type = 'Dirichlet',
                                  kernel_projection='Fourier',
                                  trainable_kernel=False)
  dict_pinns[i].gaussian_scale = 50.

  # Adding advection
  velocity = np.concatenate([[points_dict['x_eq']*0. + direction]], 1).T
  dict_pinns[i].add_coupled_variable('velocity', velocity)
  dict_pinns[i].add_advection_term('velocity')

  # Adding Power
  power = x_mesh * 0.
  power[x_mesh < agg_width[0]] = qext[0]
  power[(x_mesh >= agg_width[0]) & (x_mesh < agg_width[1])] = qext[1]
  power = tf.convert_to_tensor(power, dtype=dict_pinns[i].dtype)
  dict_pinns[i].add_coupled_variable('power', power)
  dict_pinns[i].add_external_reaction_term('power', coef=1.0, ID=0)

  # Adding self-reaction
  self_reaction_coef = x_mesh * 0. + 1.
  self_reaction_coef[x_mesh < agg_width[0]] = sigt[0]
  self_reaction_coef[(x_mesh >= agg_width[0]) & (x_mesh < agg_width[1])] = sigt[1]
  self_reaction_coef = tf.convert_to_tensor(self_reaction_coef, dtype=dict_pinns[i].dtype)
  dict_pinns[i].add_self_rection_term(self_reaction_coef)

# Scattering source iterations
iteration, max_iterations, error, tolerance = 0, 200, 1., 1e-4
current_epochs = nt_epochs
error_list, epochs_list = [], []
scalar_flux = tf.constant(0., dtype=dict_pinns[0].dtype)

while iteration <= max_iterations and error > tolerance:

  print('\n Iteration: {0} \n'.format(iteration))

  for i, _ in enumerate(mu_q):
    if iteration < 3:
      dict_pinns[i].fit(tf_epochs, coupled_optimizer, restart_model = True)
    else:
      dict_pinns[i].fit(tf_epochs, coupled_optimizer, restart_model = False)

  # Update scalar fluxes and compute error
  scalar_flux_old = scalar_flux * 1
  scalar_flux = compute_scalar_flux(dict_pinns)
  error = (tf.norm(scalar_flux - scalar_flux_old) / tf.norm(scalar_flux_old)).numpy()
  print('\n *************** \n Error: {0:.2e} \n ***************'.format(error))

  # Adaptive number of epochs
  error_list.append(error)
  epochs_list.append(current_epochs)

  if iteration >= 3:
    delta_error_current = abs(error_list[iteration] - error_list[iteration-1])
    delta_error_old = abs(error_list[iteration-1] - error_list[iteration-2])
    inflation_epochs_current = epochs_list[iteration] / epochs_list[iteration-1]
    inflation_epochs_old = epochs_list[iteration-1] / epochs_list[iteration-2]
    # Estimating current order of convergence
    error_delta = (delta_error_current) / (delta_error_old)
    error_delta -= np.log(abs(inflation_epochs_current/inflation_epochs_old))
    p = np.log(abs(error_delta)) / np.log(inflation_epochs_old + 1)
    # Estimating convergence log-ordinate
    c = abs(delta_error_current / (epochs_list[iteration-1]**p * (inflation_epochs_current**p)))
    # Estimating spectral radius
    sr = delta_error_old / delta_error_current
    # Estimating next source iteration error
    ne = abs(delta_error_current / sr)
    # Fixing new number of epochs
    current_epochs = int(max(min(min((ne / c) ** (1/p), current_epochs*1.2),1000),40))

  print('Computed new epochs {0}'.format(current_epochs))
  coupled_optimizer['nt_config'].maxIter = current_epochs

  add_scattering_sources(dict_pinns)

  iteration += 1


# In[ ]:
sigt_ = np.array([5.0, 5.0 ])
sigs_ = np.array([2.5, 2.5 ])
qext_ = np.array([5.0, 0.0 ])
width_= np.array([0.5, 0.5 ])


# ### boundary values

psi_bc = np.zeros(n_dir)


# ## run and plot

SI_max = 10000
SI_tol = 1e-8

n_ref=1000
sigt,sigs,qext,x,xm,dx = create_1d_slab_mesh_data(sigt_,sigs_,qext_,width_,n_ref)

Phi, Psi_cell = solve(sigt, sigs, qext, dx, psi_bc, aFD=0.5, SI_tol=SI_tol, SI_max=SI_max, verbose=1)

plt.figure()
plt.plot(x_mesh, predict_PiNN_dir(0, x_mesh), label='Pinn Dir. 0')
plt.plot(x_mesh, predict_PiNN_dir(1, x_mesh), label='Pinn Dir. 1')
for dir_ in range(len(mu_q)):
    plt.plot(xm, Psi_cell[:,dir_], '--', label='FD Dir. '+str(dir_))
plt.legend()
plt.grid()
plt.savefig('afPolzup2.jpg')

plt.figure()
plt.plot(x_mesh, scalar_flux, label='Pinn', linewidth=2.)
plt.plot(xm, Phi, '--', label='FD')
plt.legend()
plt.grid()
plt.savefig('sfPolzup2.jpg')

plt.figure()
plt.semilogy(error_list, label='Source Iterations Convergence')
plt.ylabel('Error')
plt.xlabel('Iterations')
plt.legend()
plt.grid()
plt.savefig('Pozulp2IterationConvergence.jpg')


# ## Reed problem with scattering

# ### S2

# In[ ]:


#############################
# Parameters
#############################
tf_epochs = 20 # Number of training epochs for the regression part
nt_epochs = 40 # Number of training epochs for the physics part
n_batches = 1 # External batches for Lagrange multipliers optimization
N_boundary = 50 # Number of samples per boundary
N_internal = 1000 # Number of internal collocation points

#############################
# Network srchitecture
#############################
dim = 1
layers = [dim] + 5*[64] + [1]

# Setting seeds
np.random.seed(17)
tf.random.set_seed(17)


##############################
# Parameters
##############################
# # Reference
sigt  = np.array([50., 5., 0., 1.,  1. ])
sigs  = np.array([ 0., 0., 0., 0.9, 0.9])
#sigs  = np.array([ 0., 0., 0., 0., 0.])
qext  = np.array([50., 0., 0., 1. , 0. ])
width = np.array([ 2., 1., 2., 1. , 2. ])

agg_width = np.cumsum(width)

snorder = 2
mu_q, w_q = np.polynomial.legendre.leggauss(snorder)
w_q /= np.sum(w_q)

##############################
# Domain
##############################
x_min, x_max = 0., 8.

# Boundaries (we create a few overlapping boundary points)
left   = np.linspace(x_min,x_min,N_boundary)
right  = np.linspace(x_max,x_max,N_boundary)

# Boundary conditions
# If Dirichlet: stores the values of the field boundary condition
# If Homogeneous or Robin: stores the values of the normals
u_bc_left  = np.array([left*0.])
u_bc_right = np.array([right*0.])

# Internal points
n_points = N_internal
x_mesh = np.linspace(x_min, x_max, n_points)

# Internal points training dict
points_dict = {}
points_dict['x_eq'] = x_mesh.flatten()

#################################
# Setting logger and optimizer
#################################
logger_pinn = {}
for i, _ in enumerate(mu_q):
  logger_pinn[i] = Logger(frequency=200)
  def error():
    return tf.reduce_sum((tf.square(dict_pinns[i].return_bc_loss()))).numpy()
  logger_pinn[i].set_error_fn(error)

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


#####################
# Creating PiNNS
#####################

def predict_PiNN_dir(i, x):
  return dict_pinns[i].predict(np.array([x]).T)

def compute_scalar_flux(dict_pinns):
  scalar_flux = np.zeros_like(x_mesh)
  for i, weight in enumerate(w_q):
    scalar_flux += predict_PiNN_dir(i, x_mesh) * weight
  return scalar_flux

def compute_scattering_vector():
  agg_width = np.cumsum(width)
  scattering_vector = x_mesh * 0.
  scattering_vector[x_mesh < agg_width[0]] = sigs[0]
  for i in range(1,len(sigs)):
    scattering_vector[(x_mesh >= agg_width[i-1]) & (x_mesh < agg_width[i])] = sigs[i]
  scattering_vector[(x_mesh >= agg_width[-1])] = sigs[-1]
  return scattering_vector

def compute_scattering_source(dict_pinns):
  return compute_scattering_vector() * compute_scalar_flux(dict_pinns) #/ 2.0

def add_scattering_sources(dict_pinns):
  scat_source = compute_scattering_source(dict_pinns)
  for i, _ in enumerate(mu_q):
    dict_pinns[i].add_coupled_variable('scat_source', scat_source)
    dict_pinns[i].add_external_reaction_term('scat_source', coef=1.0, ID=1)


dict_pinns = {}
for i, direction in enumerate(mu_q):

  #Creating PiNNs class
  if direction > 0:
    x_cord_bc = np.array([left])
    points_dict['x_bc'] = x_cord_bc.flatten()
    u_train_bc = u_bc_left.flatten()
  else:
    x_cord_bc = np.array([right])
    points_dict['x_bc'] = x_cord_bc.flatten()
    u_train_bc = u_bc_right.flatten()


  dict_pinns[i] = PhysicsInformedNN_ADR(layers=layers, optimizer=tf_optimizer, logger=logger_pinn[i], 
                                  dim = dim, points_dict=points_dict, 
                                  u_bc=u_train_bc, bc_type = 'Dirichlet',
                                  kernel_projection='Fourier',
                                  trainable_kernel=False)
  dict_pinns[i].gaussian_scale = 50.

  # Adding advection
  velocity = np.concatenate([[points_dict['x_eq']*0. + direction]], 1).T
  dict_pinns[i].add_coupled_variable('velocity', velocity)
  dict_pinns[i].add_advection_term('velocity')

  # Adding Power
  power = x_mesh * 0.
  power[x_mesh < agg_width[0]] = qext[0]
  power[(x_mesh >= agg_width[0]) & (x_mesh < agg_width[1])] = qext[1]
  power[(x_mesh >= agg_width[1]) & (x_mesh < agg_width[2])] = qext[2]
  power[(x_mesh >= agg_width[2]) & (x_mesh < agg_width[3])] = qext[3]
  power[(x_mesh >= agg_width[3])] = qext[4]
  power = tf.convert_to_tensor(power, dtype=dict_pinns[i].dtype)
  dict_pinns[i].add_coupled_variable('power', power)
  dict_pinns[i].add_external_reaction_term('power', coef=1.0, ID=0)

  # Adding self-reaction
  self_reaction_coef = x_mesh * 0. + 1.
  self_reaction_coef[x_mesh < agg_width[0]] = sigt[0]
  self_reaction_coef[(x_mesh >= agg_width[0]) & (x_mesh < agg_width[1])] = sigt[1]
  self_reaction_coef[(x_mesh >= agg_width[1]) & (x_mesh < agg_width[2])] = sigt[2]
  self_reaction_coef[(x_mesh >= agg_width[2]) & (x_mesh < agg_width[3])] = sigt[3]
  self_reaction_coef[(x_mesh >= agg_width[3])] = sigt[4]
  self_reaction_coef = tf.convert_to_tensor(self_reaction_coef, dtype=dict_pinns[i].dtype)
  dict_pinns[i].add_self_rection_term(self_reaction_coef)

# Scattering source iterations
iteration, max_iterations, error, tolerance = 0, 200, 1., 1e-4
current_epochs = nt_epochs
error_list, epochs_list = [], []
scalar_flux = tf.constant(0., dtype=dict_pinns[0].dtype)

while iteration <= max_iterations and error > tolerance:

  print('\n Iteration: {0} \n'.format(iteration))

  for i, _ in enumerate(mu_q):
    if iteration < 3:
      dict_pinns[i].fit(tf_epochs, coupled_optimizer, restart_model = True)
    else:
      dict_pinns[i].fit(tf_epochs, coupled_optimizer, restart_model = False)

  # Update scalar fluxes and compute error
  scalar_flux_old = scalar_flux * 1
  scalar_flux = compute_scalar_flux(dict_pinns)
  error = (tf.norm(scalar_flux - scalar_flux_old) / tf.norm(scalar_flux_old)).numpy()
  print('\n *************** \n Error: {0:.2e} \n ***************'.format(error))

  # Adaptive number of epochs
  error_list.append(error)
  epochs_list.append(current_epochs)

  if iteration >= 3:
    delta_error_current = abs(error_list[iteration] - error_list[iteration-1])
    delta_error_old = abs(error_list[iteration-1] - error_list[iteration-2])
    inflation_epochs_current = epochs_list[iteration] / epochs_list[iteration-1]
    inflation_epochs_old = epochs_list[iteration-1] / epochs_list[iteration-2]
    # Estimating current order of convergence
    error_delta = (delta_error_current) / (delta_error_old)
    error_delta -= np.log(abs(inflation_epochs_current/inflation_epochs_old))
    p = np.log(abs(error_delta)) / np.log(inflation_epochs_old + 1)
    # Estimating convergence log-ordinate
    c = abs(delta_error_current / (epochs_list[iteration-1]**p * (inflation_epochs_current**p)))
    # Estimating spectral radius
    sr = delta_error_old / delta_error_current
    # Estimating next source iteration error
    ne = abs(delta_error_current / sr)
    # Fixing new number of epochs
    current_epochs = int(max(min(min((ne / c) ** (1/p), current_epochs*1.2),1000),40))

  print('Computed new epochs {0}'.format(current_epochs))
  coupled_optimizer['nt_config'].maxIter = current_epochs

  add_scattering_sources(dict_pinns)

  iteration += 1


# reed
sigt_ = np.array([50., 5., 0., 1.,  1. ])
sigs_ = np.array([ 0., 0., 0., 0.9, 0.9])
qext_ = np.array([50., 0., 0., 1. , 0. ])
width_= np.array([ 2., 1., 2., 1. , 2. ])


# ### boundary values

psi_bc = np.zeros(n_dir)


# ## run and plot

SI_max = 10000
SI_tol = 1e-8

n_ref=1000
sigt,sigs,qext,x,xm,dx = create_1d_slab_mesh_data(sigt_,sigs_,qext_,width_,n_ref)

Phi, Psi_cell = solve(sigt, sigs, qext, dx, psi_bc, aFD=0.5, SI_tol=SI_tol, SI_max=SI_max, verbose=1)


# In[ ]:


plt.figure()
plt.plot(x_mesh, predict_PiNN_dir(0, x_mesh), label='Pinn Dir. 0')
plt.plot(x_mesh, predict_PiNN_dir(1, x_mesh), label='Pinn Dir. 1')
for dir_ in range(len(mu_q)):
    plt.plot(xm, Psi_cell[:,dir_], '--', label='FD Dir. '+str(dir_))
plt.legend()
plt.grid()
plt.savefig('afReed.jpg')

plt.figure()
plt.plot(x_mesh, scalar_flux, label='Pinn', linewidth=2.)
plt.plot(xm, Phi, '--', label='FD')
plt.legend()
plt.grid()
plt.savefig('sfReed.jpg')

plt.figure()
plt.semilogy(error_list, label='Source Iterations Convergence')
plt.ylabel('Error')
plt.xlabel('Iterations')
plt.legend()
plt.grid()
plt.savefig('2PiNNsIterationConvergence.jpg')


#############################
# Parameters
#############################
tf_epochs = 100 # Number of training epochs for the regression part
nt_epochs = 2000 # Number of training epochs for the physics part
n_batches = 5 # External batches for Lagrange multipliers optimization
N_boundary = 50 # Number of samples per boundary
N_internal = 1000 # Number of internal collocation points

#############################
# Network srchitecture
#############################
dim = 1
layers = [dim] + 5*[64] + [1]

# Setting seeds
np.random.seed(17)
tf.random.set_seed(17)


##############################
# Parameters
##############################
# # Reference
sigt = np.array([50., 0.0 ])
sigs = np.array([0.0, 0.0 ])
qext = np.array([5.0, 0.0 ])
width= np.array([1.0, 4.0 ])

agg_width = np.cumsum(width)

snorder = 2
mu_q, w_q = np.polynomial.legendre.leggauss(snorder)
w_q /= np.sum(w_q)

##############################
# Domain
##############################
x_min, x_max = 0., 5.

# Boundaries (we create a few overlapping boundary points)
left   = np.linspace(x_min,x_min,N_boundary)
right  = np.linspace(x_max,x_max,N_boundary)

# Boundary conditions
# If Dirichlet: stores the values of the field boundary condition
# If Homogeneous or Robin: stores the values of the normals
u_bc_left  = np.array([left*0.])
u_bc_right = np.array([right*0.])

# Internal points
n_points = N_internal
x_mesh = np.linspace(x_min, x_max, n_points)

# Internal points training dict
points_dict = {}
points_dict['x_eq'] = x_mesh.flatten()

#################################
# Setting logger and optimizer
#################################
logger_pinn = {}
for i, _ in enumerate(mu_q):
  logger_pinn[i] = Logger(frequency=20)
  def error():
    return tf.reduce_sum((tf.square(dict_pinns[i].return_bc_loss()))).numpy()
  logger_pinn[i].set_error_fn(error)

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


#####################
# Creating PiNNS
#####################

def predict_PiNN_dir(i, x):
  return dict_pinns[i].predict(np.array([x]).T)

def compute_scalar_flux(dict_pinns):
  scalar_flux = np.zeros_like(x_mesh)
  for i, weight in enumerate(w_q):
    scalar_flux += predict_PiNN_dir(i, x_mesh) * weight
  return scalar_flux


dict_pinns = {}
for i, direction in enumerate(mu_q):

  #Creating PiNNs class
  if direction > 0:
    x_cord_bc = np.array([left])
    points_dict['x_bc'] = x_cord_bc.flatten()
    u_train_bc = u_bc_left.flatten()
  else:
    x_cord_bc = np.array([right])
    points_dict['x_bc'] = x_cord_bc.flatten()
    u_train_bc = u_bc_right.flatten()


  dict_pinns[i] = PhysicsInformedNN_ADR(layers=layers, optimizer=tf_optimizer, logger=logger_pinn[i], 
                                  dim = dim, points_dict=points_dict, 
                                  u_bc=u_train_bc, bc_type = 'Dirichlet',
                                  kernel_projection='Fourier',
                                  trainable_kernel=False)
  dict_pinns[i].gaussian_scale = 50.

  # Adding advection
  velocity = np.concatenate([[points_dict['x_eq']*0. + direction]], 1).T
  dict_pinns[i].add_coupled_variable('velocity', velocity)
  dict_pinns[i].add_advection_term('velocity')

  # Adding Power
  power = x_mesh * 0.
  power[x_mesh < agg_width[0]] = qext[0]
  power[(x_mesh >= agg_width[0]) & (x_mesh < agg_width[1])] = qext[1]
  power = tf.convert_to_tensor(power, dtype=dict_pinns[i].dtype)
  dict_pinns[i].add_coupled_variable('power', power)
  dict_pinns[i].add_external_reaction_term('power', coef=1.0, ID=0)

  # Adding self-reaction
  self_reaction_coef = x_mesh * 0. + 1.
  self_reaction_coef[x_mesh < agg_width[0]] = sigt[0]
  self_reaction_coef[(x_mesh >= agg_width[0]) & (x_mesh < agg_width[1])] = sigt[1]
  self_reaction_coef = tf.convert_to_tensor(self_reaction_coef, dtype=dict_pinns[i].dtype)
  dict_pinns[i].add_self_rection_term(self_reaction_coef)

for i, _ in enumerate(mu_q):
  dict_pinns[i].fit(tf_epochs, coupled_optimizer, restart_model = True)

scalar_flux = compute_scalar_flux(dict_pinns)

sigt_ = np.array([50., 0.0 ])
sigs_ = np.array([0.0, 0.0 ])
qext_ = np.array([5.0, 0.0 ])
width_= np.array([1.0, 4.0 ])


# ### boundary values

psi_bc = np.zeros(n_dir)


# ## run and plot

SI_max = 10000
SI_tol = 1e-8

n_ref=1000
sigt,sigs,qext,x,xm,dx = create_1d_slab_mesh_data(sigt_,sigs_,qext_,width_,n_ref)

Phi, Psi_cell = solve(sigt, sigs, qext, dx, psi_bc, aFD=0.5, SI_tol=SI_tol, SI_max=SI_max, verbose=1)

# In[22]:


plt.figure()
plt.plot(x_mesh, predict_PiNN_dir(0, x_mesh), label='Pinn Dir. 0')
plt.plot(x_mesh, predict_PiNN_dir(1, x_mesh), label='Pinn Dir. 1')
for dir_ in range(len(mu_q)):
    plt.plot(xm, Psi_cell[:,dir_], '--', label='FD Dir. '+str(dir_))
plt.legend()
plt.grid()
plt.savefig('afPolzup4.jpg')

plt.figure()
plt.plot(x_mesh, scalar_flux, label='Pinn', linewidth=2.)
plt.plot(xm, Phi, '--', label='FD')
plt.legend()
plt.grid()
plt.savefig('sfPolzup4.jpg')


#############################
# Parameters
#############################
tf_epochs = 100 # Number of training epochs for the regression part
nt_epochs = 2000 # Number of training epochs for the physics part
n_batches = 5 # External batches for Lagrange multipliers optimization
N_boundary = 50 # Number of samples per boundary
N_internal = 1000 # Number of internal collocation points

#############################
# Network srchitecture
#############################
dim = 1
layers = [dim] + 5*[64] + [1]

# Setting seeds
np.random.seed(17)
tf.random.set_seed(17)


##############################
# Parameters
##############################
# # Reference

sigt = np.array([50., 1.0 ])
sigs = np.array([0.0, 0.0 ])
qext = np.array([5.0, 0.0 ])
width = np.array([1.0, 4.0 ])

agg_width = np.cumsum(width)

snorder = 2
mu_q, w_q = np.polynomial.legendre.leggauss(snorder)
w_q /= np.sum(w_q)

##############################
# Domain
##############################
x_min, x_max = 0., 5.

# Boundaries (we create a few overlapping boundary points)
left   = np.linspace(x_min,x_min,N_boundary)
right  = np.linspace(x_max,x_max,N_boundary)

# Boundary conditions
# If Dirichlet: stores the values of the field boundary condition
# If Homogeneous or Robin: stores the values of the normals
u_bc_left  = np.array([left*0.])
u_bc_right = np.array([right*0.])

# Internal points
n_points = N_internal
x_mesh = np.linspace(x_min, x_max, n_points)

# Internal points training dict
points_dict = {}
points_dict['x_eq'] = x_mesh.flatten()

#################################
# Setting logger and optimizer
#################################
logger_pinn = {}
for i, _ in enumerate(mu_q):
  logger_pinn[i] = Logger(frequency=20)
  def error():
    return tf.reduce_sum((tf.square(dict_pinns[i].return_bc_loss()))).numpy()
  logger_pinn[i].set_error_fn(error)

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


#####################
# Creating PiNNS
#####################

def predict_PiNN_dir(i, x):
  return dict_pinns[i].predict(np.array([x]).T)

def compute_scalar_flux(dict_pinns):
  scalar_flux = np.zeros_like(x_mesh)
  for i, weight in enumerate(w_q):
    scalar_flux += predict_PiNN_dir(i, x_mesh) * weight
  return scalar_flux


dict_pinns = {}
for i, direction in enumerate(mu_q):

  #Creating PiNNs class
  if direction > 0:
    x_cord_bc = np.array([left])
    points_dict['x_bc'] = x_cord_bc.flatten()
    u_train_bc = u_bc_left.flatten()
  else:
    x_cord_bc = np.array([right])
    points_dict['x_bc'] = x_cord_bc.flatten()
    u_train_bc = u_bc_right.flatten()


  dict_pinns[i] = PhysicsInformedNN_ADR(layers=layers, optimizer=tf_optimizer, logger=logger_pinn[i], 
                                  dim = dim, points_dict=points_dict, 
                                  u_bc=u_train_bc, bc_type = 'Dirichlet',
                                  kernel_projection='Fourier',
                                  trainable_kernel=False)
  dict_pinns[i].gaussian_scale = 50.

  # Adding advection
  velocity = np.concatenate([[points_dict['x_eq']*0. + direction]], 1).T
  dict_pinns[i].add_coupled_variable('velocity', velocity)
  dict_pinns[i].add_advection_term('velocity')

  # Adding Power
  power = x_mesh * 0.
  power[x_mesh < agg_width[0]] = qext[0]
  power[(x_mesh >= agg_width[0]) & (x_mesh < agg_width[1])] = qext[1]
  power = tf.convert_to_tensor(power, dtype=dict_pinns[i].dtype)
  dict_pinns[i].add_coupled_variable('power', power)
  dict_pinns[i].add_external_reaction_term('power', coef=1.0, ID=0)

  # Adding self-reaction
  self_reaction_coef = x_mesh * 0. + 1.
  self_reaction_coef[x_mesh < agg_width[0]] = sigt[0]
  self_reaction_coef[(x_mesh >= agg_width[0]) & (x_mesh < agg_width[1])] = sigt[1]
  self_reaction_coef = tf.convert_to_tensor(self_reaction_coef, dtype=dict_pinns[i].dtype)
  dict_pinns[i].add_self_rection_term(self_reaction_coef)

for i, _ in enumerate(mu_q):
  dict_pinns[i].fit(tf_epochs, coupled_optimizer, restart_model = True)

scalar_flux = compute_scalar_flux(dict_pinns)

sigt_ = np.array([50., 1.0 ])
sigs_ = np.array([0.0, 0.0 ])
qext_ = np.array([5.0, 0.0 ])
width_= np.array([1.0, 4.0 ])


# ### boundary values

psi_bc = np.zeros(n_dir)


# ## run and plot

SI_max = 10000
SI_tol = 1e-8

n_ref=1000
sigt,sigs,qext,x,xm,dx = create_1d_slab_mesh_data(sigt_,sigs_,qext_,width_,n_ref)

Phi, Psi_cell = solve(sigt, sigs, qext, dx, psi_bc, aFD=0.5, SI_tol=SI_tol, SI_max=SI_max, verbose=1)

# In[22]:


plt.figure()
plt.plot(x_mesh, predict_PiNN_dir(0, x_mesh), label='Pinn Dir. 0')
plt.plot(x_mesh, predict_PiNN_dir(1, x_mesh), label='Pinn Dir. 1')
for dir_ in range(len(mu_q)):
    plt.plot(xm, Psi_cell[:,dir_], '--', label='FD Dir. '+str(dir_))
plt.legend()
plt.grid()
plt.savefig('afPolzup5.jpg')

plt.figure()
plt.plot(x_mesh, scalar_flux, label='Pinn', linewidth=2.)
plt.plot(xm, Phi, '--', label='FD')
plt.legend()
plt.grid()
plt.savefig('sfPolzup5.jpg')

#############################
# Parameters
#############################
tf_epochs = 100 # Number of training epochs for the regression part
nt_epochs = 2000 # Number of training epochs for the physics part
n_batches = 1 # External batches for Lagrange multipliers optimization
N_boundary = 50 # Number of samples per boundary
N_internal = 1000 # Number of internal collocation points

#############################
# Network srchitecture
#############################
dim = 1
layers = [dim] + 5*[64] + [1]

# Setting seeds
np.random.seed(17)
tf.random.set_seed(17)


##############################
# Parameters
##############################
# # Reference

sigt = np.array([50., 5.0 ])
sigs = np.array([0.0, 0.0 ])
qext = np.array([5.0, 0.0 ])
width= np.array([1.0, 4.0 ])

agg_width = np.cumsum(width)

snorder = 2
mu_q, w_q = np.polynomial.legendre.leggauss(snorder)
w_q /= np.sum(w_q)

##############################
# Domain
##############################
x_min, x_max = 0., 5.

# Boundaries (we create a few overlapping boundary points)
left   = np.linspace(x_min,x_min,N_boundary)
right  = np.linspace(x_max,x_max,N_boundary)

# Boundary conditions
# If Dirichlet: stores the values of the field boundary condition
# If Homogeneous or Robin: stores the values of the normals
u_bc_left  = np.array([left*0.])
u_bc_right = np.array([right*0.])

# Internal points
n_points = N_internal
x_mesh = np.linspace(x_min, x_max, n_points)

# Internal points training dict
points_dict = {}
points_dict['x_eq'] = x_mesh.flatten()

#################################
# Setting logger and optimizer
#################################
logger_pinn = {}
for i, _ in enumerate(mu_q):
  logger_pinn[i] = Logger(frequency=20)
  def error():
    return tf.reduce_sum((tf.square(dict_pinns[i].return_bc_loss()))).numpy()
  logger_pinn[i].set_error_fn(error)

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


#####################
# Creating PiNNS
#####################

def predict_PiNN_dir(i, x):
  return dict_pinns[i].predict(np.array([x]).T)

def compute_scalar_flux(dict_pinns):
  scalar_flux = np.zeros_like(x_mesh)
  for i, weight in enumerate(w_q):
    scalar_flux += predict_PiNN_dir(i, x_mesh) * weight
  return scalar_flux


dict_pinns = {}
for i, direction in enumerate(mu_q):

  #Creating PiNNs class
  if direction > 0:
    x_cord_bc = np.array([left])
    points_dict['x_bc'] = x_cord_bc.flatten()
    u_train_bc = u_bc_left.flatten()
  else:
    x_cord_bc = np.array([right])
    points_dict['x_bc'] = x_cord_bc.flatten()
    u_train_bc = u_bc_right.flatten()


  dict_pinns[i] = PhysicsInformedNN_ADR(layers=layers, optimizer=tf_optimizer, logger=logger_pinn[i], 
                                  dim = dim, points_dict=points_dict, 
                                  u_bc=u_train_bc, bc_type = 'Dirichlet',
                                  kernel_projection='Fourier',
                                  trainable_kernel=False)
  dict_pinns[i].gaussian_scale = 50.

  # Adding advection
  velocity = np.concatenate([[points_dict['x_eq']*0. + direction]], 1).T
  dict_pinns[i].add_coupled_variable('velocity', velocity)
  dict_pinns[i].add_advection_term('velocity')

  # Adding Power
  power = x_mesh * 0.
  power[x_mesh < agg_width[0]] = qext[0]
  power[(x_mesh >= agg_width[0]) & (x_mesh < agg_width[1])] = qext[1]
  power = tf.convert_to_tensor(power, dtype=dict_pinns[i].dtype)
  dict_pinns[i].add_coupled_variable('power', power)
  dict_pinns[i].add_external_reaction_term('power', coef=1.0, ID=0)

  # Adding self-reaction
  self_reaction_coef = x_mesh * 0. + 1.
  self_reaction_coef[x_mesh < agg_width[0]] = sigt[0]
  self_reaction_coef[(x_mesh >= agg_width[0]) & (x_mesh < agg_width[1])] = sigt[1]
  self_reaction_coef = tf.convert_to_tensor(self_reaction_coef, dtype=dict_pinns[i].dtype)
  dict_pinns[i].add_self_rection_term(self_reaction_coef)

for i, _ in enumerate(mu_q):
  dict_pinns[i].fit(tf_epochs, coupled_optimizer, restart_model = True)

scalar_flux = compute_scalar_flux(dict_pinns)

sigt_ = np.array([50., 5.0 ])
sigs_ = np.array([0.0, 0.0 ])
qext_ = np.array([5.0, 0.0 ])
width_= np.array([1.0, 4.0 ])


# ### boundary values

psi_bc = np.zeros(n_dir)


# ## run and plot

SI_max = 10000
SI_tol = 1e-8

n_ref=1000
sigt,sigs,qext,x,xm,dx = create_1d_slab_mesh_data(sigt_,sigs_,qext_,width_,n_ref)

Phi, Psi_cell = solve(sigt, sigs, qext, dx, psi_bc, aFD=0.5, SI_tol=SI_tol, SI_max=SI_max, verbose=1)

# In[22]:


plt.figure()
plt.plot(x_mesh, predict_PiNN_dir(0, x_mesh), label='Pinn Dir. 0')
plt.plot(x_mesh, predict_PiNN_dir(1, x_mesh), label='Pinn Dir. 1')
for dir_ in range(len(mu_q)):
    plt.plot(xm, Psi_cell[:,dir_], '--', label='FD Dir. '+str(dir_))
plt.legend()
plt.grid()
plt.savefig('afPolzup6.jpg')

plt.figure()
plt.plot(x_mesh, scalar_flux, label='Pinn', linewidth=2.)
plt.plot(xm, Phi, '--', label='FD')
plt.legend()
plt.grid()
plt.savefig('sfPolzup6.jpg')