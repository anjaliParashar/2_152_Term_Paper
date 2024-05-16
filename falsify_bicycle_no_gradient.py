import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp
import pickle
import random
from jax.nn import logsumexp,relu
from tqdm import tqdm
from methods import SGD,QSGD,ULA
import pickle
from jax  import jacrev,hessian,vmap, random, grad
#from bicycle_2d import potential2,xgrad_compute_obs


def potential2(obs):
  X = jnp.array([[1.75  , 1.],
    [1.75     , 1.0275474],
    [1.7500226, 1.1253785],
    [1.7506951, 1.3078581],
    [1.7575585, 1.5882809],
    [1.7972672, 1.9720362],
    [1.9450787, 2.4254317],
    [2.2775965, 2.8095267],
    [2.6784875, 2.9748964],
    [2.9351635, 2.999509 ]])
  rho= 1000
  obs_d = jnp.array([jnp.linalg.norm(x-obs)**2 for x in X])
  return -logsumexp(-rho*obs_d, b=1/len(obs_d))/rho #jnp.linalg.norm(obs_d)

def h_z(z):
    origin =jnp.array([2.5,2.0])
    return jnp.linalg.norm(z-origin)**2 -100

def projection_operator(z):
    grad_h = grad(h_z,argnums=0)(z)
    proj_ = ((grad_h@grad_h.T)/jnp.linalg.norm(grad_h)**2)*relu(h_z(z)/jnp.linalg.norm(h_z(z)))
    proj = z-proj_@z
    return proj

def run_ULA(Z0, n_epochs=1,step=0.01):
    Z_list = []
    cost_list = []
    X_list = []
    Q_list = []
    Zi = Z0
    seed = 0
    for i in  tqdm(range(n_epochs)):
        grad_list = []
        x_list = []
        cost_list_ = []
        #Compute gradient for each
        for z in Zi:
            grad, X, cost_ = xgrad_compute_obs(z)
            grad_list.append(grad)
            x_list.append(X)
            cost_list_.append(cost_)
        X_list.append(jnp.array(x_list).squeeze())
        grad = jnp.array(grad_list).squeeze()
        Zi = ULA(Zi,grad,seed,step)
        Z_list.append(Zi)
        Q_list.append(jnp.sum(Zi)/p)
        cost_list.append(jnp.array(cost_list_).squeeze())
        seed+=1
    Z_list = jnp.array(Z_list).squeeze()
    return cost_list, Z_list,Q_list, X_list                                                                                                                       

def run_QSGD(Z0, n_epochs=1,step=0.1,k=1.0,p=1000):
    Z_list = [Z0]
    Q_list = [jnp.sum(Z0,axis=0)/p]
    cost_list = [potential2(Z0)]
    Zi = Z0
    seed = 0
    K = k*jnp.ones(Z0.shape[0])
    for i in  tqdm(range(n_epochs)):
        grad = vmap(jacrev(potential2))(Zi) #.squeeze(axis=2))
        Zi = QSGD(Zi,grad,seed,K,p,step)
        Zi = vmap(projection_operator)(Zi)
        Z_list.append(Zi)
        Q_list.append(jnp.sum(Zi,axis=0)/p)
        #cost_list.append(potential2(Zi)) 
        print('cost:',jnp.average(vmap(potential2)(Zi)))
        Q = jnp.sum(Zi,axis=0)/p
        print('Q:',Q)
        cost_list.append(potential2(Q))
        seed+=1
    Z_list = jnp.array(Z_list).squeeze()
    Q_list = jnp.array(Q_list).squeeze()
    cost_list = jnp.array(cost_list).squeeze()
    return cost_list, Z_list,Q_list

def run_SGD(Z0, n_epochs=1,step=0.1,k=1.0,p=1000):
    Z_list = []
    cost_list = []
    X_list = []
    Q_list = []
    Zi = Z0
    seed = 0
    K = k*jnp.ones(Z0.shape[0])
    for i in  tqdm(range(n_epochs)):
        grad_list = []
        x_list = []
        cost_list_ = []
        #Compute gradient for each
        for z in Zi:
            grad, X, cost_ = xgrad_compute_obs(z)
            grad_list.append(grad)
            x_list.append(X)
            cost_list_.append(cost_)
        X_list.append(jnp.array(x_list).squeeze())
        grad = jnp.array(grad_list).squeeze()
        Zi = QSGD(Zi,grad,seed,K,p,step)
        Z_list.append(Zi)
        Q_list.append(jnp.sum(Zi,axis=0)/p)
        print(Q_list[-1])
        cost_list.append(jnp.array(cost_list_).squeeze())
        seed+=1
    Z_list = jnp.array(Z_list).squeeze()
    return cost_list, Z_list,Q_list, X_list

p = 100
n_agents=1
d=2
seed = 2
n_epochs = 100
step=0.1

key = random.PRNGKey(seed) 
Z01 = random.uniform(key, shape=(p,n_agents,d),minval=-4,maxval=-2)
Z02 = random.uniform(key, shape=(p,n_agents,d),minval=-2,maxval=0)
Z03 = random.uniform(key, shape=(p,n_agents,d),minval=0,maxval=2)
Z04 = random.uniform(key, shape=(p,n_agents,d),minval=1,maxval=3)
Z05 = random.uniform(key, shape=(p,n_agents,d),minval=4,maxval=6)
Z06 = random.uniform(key, shape=(p,n_agents,d),minval=-6,maxval=6)

Z0list = [Z01,Z02,Z03,Z04,Z05,Z06]
#Z0list = [Z04]

cost_qsgd1, z_qsgd1, q_qsgd1 = run_QSGD(Z01, n_epochs, step=0.1, k=1.2, p=p)
cost_qsgd2, z_qsgd2, q_qsgd2 = run_QSGD(Z02, n_epochs, step=0.1, k=1.2, p=p)
cost_qsgd3, z_qsgd3, q_qsgd3 = run_QSGD(Z03, n_epochs, step=0.1, k=1.2, p=p)
cost_qsgd4, z_qsgd4, q_qsgd4 = run_QSGD(Z04, n_epochs, step=0.1, k=1.2, p=p)
cost_qsgd5, z_qsgd5, q_qsgd5 = run_QSGD(Z05, n_epochs, step=0.1, k=1.2, p=p)
cost_qsgd6, z_qsgd6, q_qsgd6 = run_QSGD(Z06, n_epochs, step=0.1, k=1.2, p=p)

data1 = {'cost':cost_qsgd1,'q':q_qsgd1,'z':z_qsgd1}
data2 = {'cost':cost_qsgd2,'q':q_qsgd2,'z':z_qsgd2}
data3 = {'cost':cost_qsgd3,'q':q_qsgd3,'z':z_qsgd3}
data4 = {'cost':cost_qsgd4,'q':q_qsgd4,'z':z_qsgd4}
data5 = {'cost':cost_qsgd5,'q':q_qsgd5,'z':z_qsgd5}
data6 = {'cost':cost_qsgd6,'q':q_qsgd6,'z':z_qsgd6}

PATH1 = '/home/anjali/work/term_paper_2_152/Bicycle_2d/SQP_bicycle_QSGD_no_grad1.pkl'
with open(PATH1, 'wb') as file1: 
    pickle.dump(data1, file1) 

PATH2 = '/home/anjali/work/term_paper_2_152/Bicycle_2d/SQP_bicycle_QSGD_no_grad2.pkl'
with open(PATH2, 'wb') as file2: 
    pickle.dump(data2, file2) 

PATH3 = '/home/anjali/work/term_paper_2_152/Bicycle_2d/SQP_bicycle_QSGD_no_grad3.pkl'
with open(PATH3, 'wb') as file3: 
    pickle.dump(data3, file3) 

PATH4 = '/home/anjali/work/term_paper_2_152/Bicycle_2d/SQP_bicycle_QSGD_no_grad4.pkl'
with open(PATH4, 'wb') as file4: 
    pickle.dump(data4, file4) 

PATH5 = '/home/anjali/work/term_paper_2_152/Bicycle_2d/SQP_bicycle_QSGD_no_grad5.pkl'
with open(PATH5, 'wb') as file5: 
    pickle.dump(data5, file5) 

PATH6 = '/home/anjali/work/term_paper_2_152/Bicycle_2d/SQP_bicycle_QSGD_no_grad6.pkl'
with open(PATH6, 'wb') as file6: 
    pickle.dump(data6, file6)

#cost_ula, Z_ula = run_ULA(Z0, n_epochs=n_epochs,step=0.01)
#print(cost_ula.shape, Z_ula.shape)
#X,x_grad = xgrad_compute(obstacle_new_)
#obstacle_new = tuple_to_tensor(obstacle_gen(d)) 
#obstacle_new = torch.tensor([[-2.5,-15.0],[-10.0,-10.0]])
#obstacle_new1 = obstacle_new[:,0:2].to(device).float()
#obs = torch.tensor(np.random.uniform(low=[-20,-20], high=[0,20], size=(2,2)))


#PATH = 'experiments/ULA/SQP_unicycle_'+N+epochs+ '_.pkl'
#data = {'cost':u_ulmc1,'samples':samples_ulmc1,'X':X_ulmc}
#with open(PATH, 'wb') as file: 
#    pickle.dump(data, file) 
"""
cost = {'c_ula':u_ula1,'c_ulmc':u_ulmc1}
data = {'d_ula':samples_ula1,'d_ulmc':samples_ulmc1}
X_list = {'X_ula':X_ula,'X_ulmc':X_ulmc}
with open('SQP_cost_200.pkl', 'wb') as file_m: 
    pickle.dump(cost, file_m) 
with open('SQP_data_200.pkl', 'wb') as file_n: 
    pickle.dump(data, file_n) 
with open('SQP_X_200.pkl', 'wb') as file_o: 
    pickle.dump(X_list, file_o) 
"""
#ulmc_np = np.array(samples_ulmc2[-2])
#ula_np = np.array(samples_ula2[-1])
 

#ulmc_np = np.reshape(ulmc_np,(ulmc_np.shape[0]*ulmc_np.shape[1],2))
#ula_np = ula_np.reshape((ula_np.shape[0]*ula_np.shape[1],2))

#ulmc_np = ulmc_np.reshape((ulmc_np.shape[0],2))
#ula_np = ula_np.reshape((ula_np.shape[0],2))
#plt.figure()
#plt.plot(u_ula,label='ula')
#plt.plot(u_ulmc,label='ulmc')
#plt.legend()
#plt.savefig('results/SQP/cost_ula_ulmc.png')
"""
plt.figure()
plot_decision_boundary(model_0, X_test, y_test)
ref_curve_ = ref_curve.cpu()
track_upper_ = track_upper.cpu().detach()
plt.plot(track_upper_[:,0],track_upper_[:,1])
plt.scatter(ref_curve_[:,0],ref_curve_[:,1])
plt.scatter(ulmc_np[:,0],ulmc_np[:,1],label='ULMC')
#plt.scatter(ula_np[:,0],ula_np[:,1],label='ULA')
plt.legend()
plt.title('ULMC vs ULA')
"""

#render_scene(ulmc_np,track_upper_,track_lower_,X.detach().cpu(),ref_curve.cpu())
#render_scene(ula_np,track_upper_,track_lower_,X.detach().cpu(),ref_curve.cpu())