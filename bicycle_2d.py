import torch
import jax.numpy as jnp
import jax
from jax import jacrev,grad
from jax.nn import logsumexp
import matplotlib.pyplot as plt
from importlib import reload
import numpy as np
from trajax import integrators
from trajax.experimental.sqp import shootsqp, util

reload(shootsqp)
reload(util)

class Dynamics: 
    def __init__(self,n,m,T,R,R_t,Q_T): 
        self.n = n
        self.m = m
        self.T = T
        self.R = R
        self.R_t = R_t
        self.Q_T = Q_T
    # getter method 
    def get_T(self): 
        return self.T
    
    def get_n(self): 
        return self.n
    
    def get_m(self): 
        return self.m
    
    def get_R(self): 
        return self.R
    
    def get_Q_T(self): 
        return self.Q_T
    
    def get_R_t(self): 
        return self.R_t

def obstacle_gen():
  obs = []
  #for i in range(len):
    #key = jax.random.PRNGKey(i)
    #obs_o = (jnp.array(np.random.uniform(low=-10, high=10, size=(2))),np.random.uniform(low=1, high=2)) #Unnormalise
    #obs_o = (jnp.array(np.random.uniform(low=-1, high=1, size=(2))),np.random.uniform(low=0.05, high=0.2)) #Normalise
  #obs_o1 = (jnp.array(np.array([-8.13855052,  5.67300081])),np.array(1.0))
  #
  #obs_o1 = (jnp.array(np.array([-13.567037, -5.4539275])),jnp.array(1.0))
  #obs_o =  (jnp.array(np.array([-2.543676, 12.797775])),jnp.array(1.0))
  obs_o =  jnp.array([1.7210718, 1.2277586])
    #obs_o = (jnp.array(np.array([-15.0,  10.0])),np.array(1.5))
    #obs.append(obs_o)
  #obs_o = (jnp.array(np.array([-10.13855052,  8.67300081])),np.array(1.0))
  obs.append(obs_o)
  return obs

def render_scene(obs):
  # Setup obstacle environment for state constraint
  #world_range = (jnp.array([-1, -1]), jnp.array([1, 1])) #Normalized
  world_range = (jnp.array([-20, -20]), jnp.array([20, 20])) #Unnormalized
  fig = plt.figure(figsize=(6,6))
  ax = fig.add_subplot(111)
  plt.grid(True)
  for ob in obs:
    ax.add_patch(plt.Circle(ob[0],ob[1],color='k', alpha=0.3))
  ax.set_xlim([world_range[0][0], world_range[1][0]])
  ax.set_ylim([world_range[0][1], world_range[1][1]])
  ax.set_aspect('equal')
  return fig, ax

def ref_specs():
  #Ref path
  #goal_default = jnp.array([0,-15, 0.,0.]) #element third was pi/2 originally
  goal_default = jnp.array([3., 3., jnp.pi/2, 0.])
  return goal_default

def cost(x, u, t):
  goal = ref_specs()
  R_t = single_car.get_R_t()
  R = single_car.get_R()
  Q_T = single_car.get_Q_T()

  stage_cost =  dt * jnp.vdot(u, R @ u)
  delta = state_wrap(x - goal)
  term_cost = jnp.vdot(delta, Q_T @ delta)
  return jnp.where(t == T, term_cost, stage_cost)

# Obstacle avoidance constraint function
def obs_constraint(pos):
  def avoid_obs(pos_c, ob):
    delta_body = pos_c - ob
    delta_dist_sq = jnp.vdot(delta_body, delta_body) - ((0.1)**2)
    return delta_dist_sq
  return jnp.array([avoid_obs(pos, ob) for ob in obs])

# State constraint function
#@jax.jit
def state_constraint(x, t):
  del t
  pos = x[0:2]
  return obs_constraint(pos)

def car_ode_bicycle(x, u, t):
  wheelbase = 2.96
  del t
  return jnp.array([x[3] * jnp.sin(x[2]),
                    x[3] * jnp.cos(x[2]),
                    x[3] * u[0],#jnp.tan(u[0])/wheelbase,#jnp.tan(u[0])/wheelbase,
                    u[1]])

#Single integrator model
# Constants
n, m, T = (4, 2, 99)
R = jnp.diag(jnp.array([0.2, 0.1]))
Q_T = jnp.diag(jnp.array([50., 50., 50., 10.])) #50., 50., 50., 10.
R_t = jnp.diag(jnp.array([2, 1])) #0.2, 0.1

def traj_opt(obs_):
  global obs
  obs=obs_

  #Dynamics
  global single_car,dt
  single_car = Dynamics(n,m,T,R,R_t,Q_T) 
  dt = 0.05

  # Setup discrete-time dynamics
  dynamics = integrators.euler(car_ode_bicycle, dt=dt)

  global state_wrap
  s1_indices = (2,) # Indices of state corresponding to S1 sphere constraints
  state_wrap = util.get_s1_wrapper(s1_indices)

  # Control box bounds 
  control_bounds = (jnp.array([-jnp.pi/3., -6.]),
                  jnp.array([jnp.pi/3., 6.]))
  #control_bounds = (jnp.array([-jnp.pi/10,-2.0]), #-2, -jnp.pi/3
  #                  jnp.array([jnp.pi/10,2.0])) #2, jnp.pi/3
 
  # Define Solver
  solver_options = dict(method=shootsqp.SQP_METHOD.SENS,
                        ddp_options={'ddp_gamma': 1e-4,'ddp_gamma_ratio': 1.0},
                        hess="full", verbose=False,
                        max_iter=200, ls_eta=0.49, ls_beta=0.8,
                        primal_tol=1e-3, dual_tol=1e-3, stall_check="abs",
                        debug=False,do_log=False)#,qp_solver=shootsqp.QP_SOLVER.QP_ALILQR)

  solver = shootsqp.ShootSQP(n, m, T, dynamics, cost, control_bounds,
                            state_constraint, s1_ind=s1_indices, **solver_options)
  
  # Set initial conditions and problem parameters
  global goal_default
  goal_default = ref_specs()

  #x0 = jnp.array([0.0,15.0,3*jnp.pi/2,0.]) 
  x0 = jnp.array([1.75, 1.0, 0., 0.])
  U0 = jnp.zeros((T, m))
  X0 = None
  solver.opt.proj_init = False
  #X0 = jnp.hstack((ref_curve, jnp.zeros((T+1, 2))))
  
  solver.opt.max_iter = 1
  _ = solver.solve(x0, U0, X0)

  # Run to completion
  solver.opt.max_iter = 100
  soln = solver.solve(x0, U0, X0)

  U, X = soln.primals
  history = soln.history
  del solver
  del obs
  
  X_ = X[:,0:2]
  U_ = U

  return U,X[:,0:2]#,potential2(X_,ref_curve[:,0:2]) #,X_,history

def render_scene(obs,X,X_unproj):
    # Setup obstacle environment for state constraint
    world_range = (jnp.array([1, 0]), jnp.array([4, 4]))

    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111)
    plt.grid(False)

    #for ob in obs:
    print(obs)
    ax.add_patch(plt.Circle([obs[0],obs[1]], radius=0.1, color='k', alpha=0.3))
    ax.set_xlim([world_range[0][0], world_range[1][0]])
    ax.set_ylim([world_range[0][1], world_range[1][1]])

    ax.set_aspect('equal')
    # Start
    ax.add_patch(plt.Circle([1.75, 1.0], 0.1, color='g', alpha=0.3))
    # End
    ax.add_patch(plt.Circle([3.0,3.0], 0.1, color='r', alpha=0.3))
    ax.plot(X[:, 0], X[:, 1], 'r-', linewidth=2,label=' Trajectory with obstacle')
    ax.plot(X_unproj[:, 0], X_unproj[:, 1], 'b-', linewidth=1,label='Nominal Trajectory')
    ax.set_aspect('equal')
    ax.set_xlabel('X',fontsize=15)
    ax.set_ylabel('Y',fontsize=15)
    ax.legend(fontsize=15)
    return fig, ax

def slack_calc(pos,obs):
    def slack(pos_c,ob):
        r = 0.5
        x0 = ob[0] #ob[0][0]
        y0 = ob[1] #ob[0][1]

        x = pos_c[0]
        y = pos_c[1]
        s_sq = (x-x0)**2 + (y-y0)**2 - (r**2)
        return s_sq
    return jnp.array([[slack(pos_i, ob) for pos_i in pos] for ob in obs])

def g_func(X,obs):
   pos = X[:,0:2]
   slack_star = X[:,2:3]
   return jnp.transpose(slack_calc(pos,obs))-slack_star

def predict_dict(params):
    return g_func(params['X'], params['obs'])

def xgrad_compute_obs_2(obs):
  U,X_= traj_opt(obs)
  cost = potential2(X_,obs)
  grad_value = grad(potential2,argnums=1)(X_,obs)
  return grad_value,X_,cost

def xgrad_compute_obs_1(obs):
    U,X_= traj_opt(obs)
    d = len(obs)
    T = 99
    obs_jnp = obs
    cost = potential2(X_,obs)
    X_gradient = grad(potential2,argnums=0)(X_,obs)
    slack_star = slack_calc(X_,obs_jnp)
    X_cat = jnp.hstack([X_,jnp.transpose(slack_star)])
    J_dict = jacrev(predict_dict)({'X': X_cat, 'obs': obs_jnp})
    X_grad = J_dict['X']
    obs_grad = J_dict['obs'].reshape((T+1,d,d,2))
    g_inv = jnp.linalg.pinv(X_grad).reshape((T+1,d+2,T+1,d))
    x_grad = -jnp.tensordot(g_inv,obs_grad)[:,0:2,:,0:2]
    grad_value = jnp.tensordot(X_gradient,x_grad) 
    #obs_gradient = torch.from_numpy(np.asarray(grad_value)).cuda()
    #X_ = np.asarray(X_)
    return grad_value,X_,cost

def potential2(X,obs):
  obs_d = jnp.array([(jnp.linalg.norm(X-ob))**2 for ob in obs])
  return logsumexp(obs_d) #jnp.linalg.norm(obs_d)


#def potential2(X,Xref):
  #obs_d = jnp.array([(jnp.linalg.norm(X-ob)) for ob in obs])
  #ref_d = jnp.linalg.norm(X-Xref)
  #return jax.nn.softplus(-ref_d+60) #jnp.linalg.norm(obs_d) - ref_d#logsumexp(rho*d,axis=0)/rho#jnp.linalg.norm(d)#-logsumexp(-rho*d,0)/rho#-logsumexp(-rho*d,0)/rho

"""
def gradient_comp(obs):
  U,X,_ = traj_opt(obs)
  cost = potential2(X,obs)
  return cost

def grad_show(obs,X,ref_curve):
  g1 = jax.jacrev(traj_opt,argnums=0)(obs)
  g2 = jax.jacrev(potential2,argnums=0)(X,ref_curve)
  grad_ = g2@g1
  return torch.from_numpy(np.asarray(grad_)).cuda()
"""

if False:
  #obs = obstacle_gen(1)
  #obs = jnp.array([[-13.567037, -5.4539275],[-2.543676, 12.797775]])
  #obs = jnp.array([[-5.8661723,13.763039]])
  obs = jnp.array([[-12.5,10],[0.0,0.0]])
  _,ref_curve = ref_specs()
  #U,X =traj_opt(obs)
  batched_traj = jax.vmap(traj_opt,in_axes=0)
  print(batched_traj(obs))
  #obs_grad,X,cost = xgrad_compute_obs(obs,ref_curve[:,0:2])
  #print(obs_grad,cost)
  #obs_grad = jax.jacrev(gradient_comp,argnums=0)(obs)
  #obs_gradient,X,history,cost = xgrad_compute_obs(obs)
  plt.plot(ref_curve[:,0],ref_curve[:,1],label='ref')
  plt.plot(X[:,0],X[:,1],label='X')
  plt.scatter(obs[0,0],obs[0,1])
  plt.scatter(obs[1,0],obs[1,1])
  plt.legend()
  plt.show()
  print(U)

  #plt.savefig('Unicycle_reverse_limits.png')

def render_scene(obs,X,ref_curve):
  # Setup obstacle environment for state constraint
    track_upper = get_track_upper(m=17)[0:100,:]
    track_lower = get_track_lower(m=12)[0:100,:]
    world_range = (np.array([-1, -1]), np.array([1, 1])) #Normalized
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111)
    plt.grid(True)
    for ob in obs:
        ax.add_patch(plt.Circle([ob[0],ob[1]],0.1,color='k', alpha=0.3))
        ax.set_xlim([world_range[0][0], world_range[1][0]])
        ax.set_ylim([world_range[0][1], world_range[1][1]])
    #ax.set_aspect('equal','box')
    plt.plot(track_upper[:,0],track_upper[:,1],label='Track Max')
    plt.plot(track_lower[:,0],track_lower[:,1],label='Track Min')
    plt.plot(X[:,0],X[:,1],'.',label='X')
    plt.plot(ref_curve[:,0],ref_curve[:,1],label='Reference')
    plt.legend(fontsize="10", loc ="upper right")
    plt.show()
    return fig, ax

#render_scene(obs,X,ref_curve)
"""
fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111)
ax.grid(True)
plt.plot(solver._timesteps[:-1]*dt, U, markersize=5)
ax.set_ylabel('U')
ax.set_xlabel('Time [s]')
"""

"""
import seaborn as sns
colors = sns.color_palette("tab10")

history = soln.history
plt.figure()
plt.rcParams.update({'font.size': 24})
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

fig, axs = plt.subplots(2, 2, figsize=(15, 15))

axs[0][0].plot(history['steplength'], color=colors[0], linewidth=2)
axs[0][0].set_title('Step size')
axs[0][0].grid(True)

axs[0][1].plot(history['obj'], color=colors[0], linewidth=2)
axs[0][1].set_title('Objective')
axs[0][1].set_yscale('log')
axs[0][1].grid(True)

axs[1][0].plot(history['min_viol'], color=colors[0], linewidth=2)
axs[1][0].set_title('Min constraint viol.')
axs[1][0].set_xlabel('Iteration')
axs[1][0].grid(True)

if 'ddp_err' in history:
  axs[1][1].plot(history['ddp_err'], color=colors[0], linewidth=2)
  axs2 = axs[1][1].twinx()
  axs2.plot(history['ddp_err_grad'], color=colors[1], linewidth=2)
  axs2.set_yscale("log")
  axs[1][1].set_title('DDP errors')
  axs[1][1].set_xlabel('Iteration')
  axs[1][1].grid(True)
"""