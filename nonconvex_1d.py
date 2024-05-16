import jax
import jax.numpy as jnp
from methods import ULA, QSGD, SGD
from tqdm import tqdm
from jax  import jacrev,hessian,vmap, random
import matplotlib.pyplot as plt
import seaborn as sns
from jax.nn import relu
#1-D cost funciton: dummy
def cost(z):
    F = 150
    f_z = (jnp.power(z,4) - 4*jnp.power(z,2) + 0.2*z + 1.2*jnp.sin(20*z) -3.5*jnp.sin(2*jnp.pi*z) + jnp.cos(10*z))/F
    return relu(f_z+0.04)

def run_ULA(Z0, cost, p,n_epochs,step):
    Z_list = [Z0]
    cost_list = [cost(Z0)]
    Q_list = [jnp.sum(Z0,axis=0)/p]
    Zi = Z0
    seed = 0
    for i in  tqdm(range(n_epochs)):
        grad = vmap(jacrev(cost))(Zi).squeeze(axis=2)
        Zi = ULA(Zi, grad, seed, step=step)
        cost_list.append(cost(Zi))
        Q_list.append(jnp.sum(Zi,axis=0)/p)
        Z_list.append(Zi)
        seed+=0
    return cost_list, Z_list, Q_list

def run_QSGD(Z0,cost, n_epochs=1,step=0.1,k=1.0,p=1000):
    Z_list = [Z0]
    Q_list = [jnp.sum(Z0,axis=0)/p]
    cost_list = [cost(Z0)]
    Zi = Z0
    seed = 0
    K = k*jnp.ones((Z0.shape[0]))
    for i in  tqdm(range(n_epochs)):
        grad = vmap(jacrev(cost))(Zi).squeeze(axis=2)
        #hess = vmap(hessian(cost))(Zi).squeeze(axis=1)
        #for j in range(Z0.shape[0]):
        #    if -hess[j]-k>0:
        #        K.at[j].set(-hess[j].squeeze()-2.5)
        Zi = QSGD(Zi,grad,seed,K,p,step)
        Z_list.append(Zi)
        Q_list.append(jnp.sum(Zi,axis=0)/p)
        cost_list.append(cost(Zi))
        seed+=1
    Z_list = jnp.array(Z_list).squeeze()
    return cost_list, Z_list,Q_list

def run_SGD(Z0,cost, n_epochs=1,step=0.1,k=1.0,p=1000):
    Z_list = [Z0]
    Q_list = [jnp.sum(Z0,axis=0)/p]
    cost_list = [cost(Z0)]
    Zi = Z0
    seed = 0
    for i in  tqdm(range(n_epochs)):
        grad = vmap(jacrev(cost))(Zi).squeeze(axis=2)
        Zi = SGD(Zi,grad,seed,k,p,step)
        Z_list.append(Zi)
        Q_list.append(jnp.sum(Zi,axis=0)/p)
        cost_list.append(cost(Zi))
        seed+=1
    return cost_list, Z_list,Q_list

key = 'compare'
if key=="na":
    p = 1000
    d=1
    seed = 2
    n_epochs = 200
    key = random.PRNGKey(seed) 
    Z0 = random.uniform(key, shape=(p,d),minval=-3,maxval=3)
    cost_qsgd, Z_qsgd, Q_qsgd = run_QSGD(Z0, cost, n_epochs=n_epochs,step=0.1,k=0.1,p=p)

    plt.plot(Q_qsgd)
    plt.show()
if key=='compare':
    p = 1000
    d=1
    seed = 2
    n_epochs = 500
    key = random.PRNGKey(seed) 
    Z0 = random.uniform(key, shape=(p,d),minval=-7,maxval=3)
    cost_qsgd, Z_qsgd, Q_qsgd = run_QSGD(Z0, cost, n_epochs=n_epochs,step=0.1,k=0.1,p=p)
    cost_ula, Z_ula, Q_ula = run_ULA(Z0, cost, p=p,n_epochs=n_epochs,step=0.01)
    cost_sgd, Z_sgd, Q_sgd = run_SGD(Z0, cost, n_epochs=n_epochs,step=0.1,k=0.1,p=p)
    Z_ula = jnp.array(Z_ula).squeeze()
    Z_sgd = jnp.array(Z_sgd).squeeze()
    Z_ula = jnp.array(Z_ula).squeeze()

    #Compare between baselines
    plt.plot(Q_sgd,linewidth=2,color='blue',label='SGD')
    plt.plot(Q_ula,linewidth=2,color='red',label='ULA')
    plt.plot(Q_qsgd,linewidth=2,color='green',label='QSGD')
    plt.hlines(-1.7,xmin=0,xmax=n_epochs,color='black',linestyles='--',label='Optimal value')
    plt.ylabel('Iterates')
    plt.title('Comparison between Langevin, SGD, Q-SGD')
    plt.xlabel("Epochs")
    plt.legend()
    plt.show()

if key=='initialize':
    #Visualization of QSGD for different initializations
    p = 1000
    d=1
    seed = 2
    n_epochs = 200
    key = random.PRNGKey(seed) 
    Z01 = random.uniform(key, shape=(p,d),minval=-7,maxval=3)
    Z02 = random.uniform(key, shape=(p,d),minval=-3,maxval=3)
    Z03 = random.uniform(key, shape=(p,d),minval=-10,maxval=-7)
    Z04 = random.uniform(key, shape=(p,d),minval=-1.5,maxval=1.5)
    Z05 = random.uniform(key, shape=(p,d),minval=0,maxval=3)
    _, _, Q_qsgd1 = run_QSGD(Z01, cost, n_epochs=n_epochs,step=0.1,k=0.1,p=p)
    _, _, Q_qsgd2= run_QSGD(Z02, cost, n_epochs=n_epochs,step=0.1,k=0.1,p=p)
    _, _, Q_qsgd3 = run_QSGD(Z03, cost, n_epochs=n_epochs,step=0.1,k=0.1,p=p)
    _, _, Q_qsgd4 = run_QSGD(Z04, cost, n_epochs=n_epochs,step=0.1,k=0.1,p=p)
    _, _, Q_qsgd5 = run_QSGD(Z05, cost, n_epochs=n_epochs,step=0.1,k=0.1,p=p)
    print(Q_qsgd1[-1],Q_qsgd2[-1],Q_qsgd3[-1],Q_qsgd4[-1],Q_qsgd5[-1])

    #Compare between baselines
    plt.plot(Q_qsgd1,linewidth=2, color='blue',label='[-7,3]')
    plt.plot(Q_qsgd2,linewidth=2,color='deeppink',label='[-3,3]')
    plt.plot(Q_qsgd3,linewidth=2,color='green',label='[-10,-7]')
    plt.plot(Q_qsgd4,linewidth=2,color='orange',label='[-1.5,1.5]')
    plt.plot(Q_qsgd5,linewidth=2,color='red',label='[0,3]')
    plt.hlines(-1.7,xmin=0,xmax=n_epochs,color='black',linestyles='--',label='Optimal value')
    plt.ylabel('Iterates')
    plt.title('Sensitivity to intialization')
    plt.xlabel("Epochs")
    plt.legend()
    plt.show()

if key=='perform':
    #Visualization of QSGD for different initializations
    p = 100
    d=1
    seed = 2
    n_epochs = 200
    key = random.PRNGKey(seed) 
    Z01 = random.uniform(key, shape=(p,d),minval=-3,maxval=-2)
    Z02 = random.uniform(key, shape=(p,d),minval=-2,maxval=-1)
    Z03 = random.uniform(key, shape=(p,d),minval=-1,maxval=0)
    Z04 = random.uniform(key, shape=(p,d),minval=0,maxval=1)
    Z05 = random.uniform(key, shape=(p,d),minval=1,maxval=2)
    Z06 = random.uniform(key, shape=(p,d),minval=2,maxval=3)
    _, Z_qsgd1, Q_qsgd1 = run_QSGD(Z01, cost, n_epochs=n_epochs,step=0.1,k=0.1,p=p)
    _, Z_qsgd2, Q_qsgd2= run_QSGD(Z02, cost, n_epochs=n_epochs,step=0.1,k=0.1,p=p)
    _, Z_qsgd3, Q_qsgd3 = run_QSGD(Z03, cost, n_epochs=n_epochs,step=0.1,k=0.1,p=p)
    _, Z_qsgd4, Q_qsgd4 = run_QSGD(Z04, cost, n_epochs=n_epochs,step=0.1,k=0.1,p=p)
    _, Z_qsgd5, Q_qsgd5 = run_QSGD(Z05, cost, n_epochs=n_epochs,step=0.1,k=0.1,p=p)
    _, Z_qsgd6, Q_qsgd6 = run_QSGD(Z06, cost, n_epochs=n_epochs,step=0.1,k=0.1,p=p)
    print(Q_qsgd1[-1],Q_qsgd2[-1],Q_qsgd3[-1],Q_qsgd4[-1],Q_qsgd5[-1],Q_qsgd6[-1])

    #Compare between baselines
    fig,axs = plt.subplots(2,1)
    axs[0].plot(Q_qsgd1,linewidth=2, color='blue',label='[-3,-2]')
    axs[0].plot(Q_qsgd2,linewidth=2,color='deeppink',label='[-2,-1]')
    axs[0].plot(Q_qsgd3,linewidth=2,color='green',label='[-1,0]')
    axs[0].plot(Q_qsgd4,linewidth=2,color='orange',label='[0,1]')
    axs[0].plot(Q_qsgd5,linewidth=2,color='red',label='[1,2]')
    axs[0].plot(Q_qsgd6,linewidth=2,color='cyan',label='[2,3]')

    axs[1].plot(Z_qsgd1,linewidth=0.5, alpha=0.2,color='blue',label='[-3,-2]')
    axs[1].plot(Z_qsgd2,linewidth=0.5,alpha=0.2,color='deeppink',label='[-2,-1]')
    axs[1].plot(Z_qsgd3,linewidth=0.5,alpha=0.2,color='green',label='[-1,0]')
    axs[1].plot(Z_qsgd4,linewidth=0.5,alpha=0.2,color='orange',label='[0,1]')
    axs[1].plot(Z_qsgd5,linewidth=0.5,alpha=0.2,color='red',label='[1,2]')
    axs[1].plot(Z_qsgd6,linewidth=0.5,alpha=0.2,color='cyan',label='[2,3]')

    #axs[0].hlines(-1.7,xmin=0,xmax=n_epochs,color='black',linestyles='--',label='Optimal value')
   # plt.ylabel('Iterates')
    #plt.title('Sensitivity to intialization')
    #plt.xlabel("Epochs")
    axs[0].legend()
    #axs[1].legend()
    plt.show()


if key =='cost_function':
    n_epochs = 500
    #Plot the cost function to observe the global and local minima
    z_lin = jnp.linspace(-3,3,n_epochs)
    cost_lin = cost(z_lin)
    plt.plot(z_lin, cost_lin, linewidth=2.5)
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('Motivating example: Modified cost function')
    plt.legend()
    plt.show()