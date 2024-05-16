import jax.numpy as jnp
import matplotlib.pyplot as plt
import pickle

def render_scene(obs,X):
    # Setup obstacle environment for state constraint
    world_range = (jnp.array([-1, -1]), jnp.array([5, 5]))

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
    ax.plot(X[:, 0], X[:, 1], 'r-', linewidth=2)
    ax.set_aspect('equal')
    return fig, ax

#Unzip pickle file
file_qsgd1 = open('/home/anjali/work/term_paper_2_152/Bicycle_2d/SQP_bicycle_QSGD_no_grad1.pkl', 'rb')
data_qsgd1 = pickle.load(file_qsgd1)
file_qsgd1.close()

file_qsgd2 = open('/home/anjali/work/term_paper_2_152/Bicycle_2d/SQP_bicycle_QSGD_no_grad2.pkl', 'rb')
data_qsgd2 = pickle.load(file_qsgd2)
file_qsgd2.close()

file_qsgd3 = open('/home/anjali/work/term_paper_2_152/Bicycle_2d/SQP_bicycle_QSGD_no_grad3.pkl', 'rb')
data_qsgd3 = pickle.load(file_qsgd3)
file_qsgd3.close()

file_qsgd4 = open('/home/anjali/work/term_paper_2_152/Bicycle_2d/SQP_bicycle_QSGD_no_grad4.pkl', 'rb')
data_qsgd4 = pickle.load(file_qsgd4)
file_qsgd4.close()

file_qsgd5 = open('/home/anjali/work/term_paper_2_152/Bicycle_2d/SQP_bicycle_QSGD_no_grad5.pkl', 'rb')
data_qsgd5 = pickle.load(file_qsgd5)
file_qsgd5.close()

file_qsgd6 = open('/home/anjali/work/term_paper_2_152/Bicycle_2d/SQP_bicycle_QSGD_no_grad6.pkl', 'rb')
data_qsgd6 = pickle.load(file_qsgd6)
file_qsgd6.close()

file_qsgd_proj1 = open('/home/anjali/work/term_paper_2_152/Bicycle_2d/SQP_bicycle_QSGD_no_grad_1.pkl', 'rb')
data_qsgd1_proj1 = pickle.load(file_qsgd_proj1)
file_qsgd_proj1.close()

file_qsgd_proj2 = open('/home/anjali/work/term_paper_2_152/Bicycle_2d/SQP_bicycle_QSGD_no_grad_extra2.5.pkl', 'rb')
data_qsgd1_proj2 = pickle.load(file_qsgd_proj2)
file_qsgd_proj2.close()

file_qsgd_proj5 = open('/home/anjali/work/term_paper_2_152/Bicycle_2d/SQP_bicycle_QSGD_no_grad_5.pkl', 'rb')
data_qsgd1_proj5 = pickle.load(file_qsgd_proj5)
file_qsgd_proj5.close()

file_qsgd_proj10 = open('/home/anjali/work/term_paper_2_152/Bicycle_2d/SQP_bicycle_QSGD_no_grad_10.pkl', 'rb')
data_qsgd1_proj10 = pickle.load(file_qsgd_proj10)
file_qsgd_proj10.close()

file_qsgd_proj100 = open('/home/anjali/work/term_paper_2_152/Bicycle_2d/SQP_bicycle_QSGD_no_grad_100.pkl', 'rb')
data_qsgd1_proj100 = pickle.load(file_qsgd_proj100)
file_qsgd_proj100.close()

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

fig = plt.figure(figsize=(6,6))
ax = fig.add_subplot(111)
plt.grid(True)
ax.add_patch(plt.Circle([0,0], radius=2.5, color='grey', alpha=0.3))
ax.scatter(data_qsgd4['z'][-1,:,0],data_qsgd4['z'][-1,:,1],color='red',linewidth=0.1,alpha=0.5)
ax.scatter(data_qsgd1_proj2['z'][-10,:,0],data_qsgd1_proj2['z'][-10,:,1],linewidth=0.1,color='cyan',alpha=0.6)
# Start
ax.add_patch(plt.Circle([1.75, 1.0], 0.1, color='g', alpha=0.3))
# End
ax.add_patch(plt.Circle([3.0,3.0], 0.1, color='r', alpha=0.3))
ax.add_patch(plt.Circle([data_qsgd1_proj2['q'][-1][0],data_qsgd1_proj2['q'][-1][1]], radius=0.1, color='cyan', alpha=0.6))
ax.add_patch(plt.Circle([data_qsgd4['q'][-1][0],data_qsgd4['q'][-1][1]], radius=0.1, color='red', alpha=0.6))

ax.plot(X[:, 0], X[:, 1], '-.', color='black',linewidth=0.5)

key='proj'
#cost analysis
if key=='cost':
    plt.figure()
    plt.plot(data_qsgd1['cost'],color='orange',label='[-4,-2]')
    plt.plot(data_qsgd2['cost'],color='blue',label='[-2,0]')
    plt.plot(data_qsgd3['cost'],color='deeppink',label='[0,2]')
    plt.plot(data_qsgd4['cost'],color='green',label='[2,4]')
    plt.plot(data_qsgd5['cost'],color='red',label='[4,6]')
    plt.plot(data_qsgd6['cost'],color='black',label='[-6,6]')
    plt.xlabel('Epochs',fontsize=15)
    plt.ylabel('$U(z)$',fontsize=15)
    plt.legend()

#Convergence of parameters
if key=='convergence':
    plt.figure()
    plt.plot(data_qsgd1['z'][:,:,0],linewidth=0.5,color='yellow',alpha=0.1)
    plt.plot(data_qsgd2['z'][:,:,0],linewidth=0.5,color='cyan',alpha=0.1)
    plt.plot(data_qsgd3['z'][:,:,0],linewidth=0.5,color='deeppink',alpha=0.1)
    plt.plot(data_qsgd4['z'][:,:,0],linewidth=0.5,color='green',alpha=0.1)
    plt.plot(data_qsgd5['z'][:,:,0],linewidth=0.5,color='red',alpha=0.1)
    plt.plot(data_qsgd6['z'][:,:,0],linewidth=0.5,color='black',alpha=0.1)

    plt.plot(data_qsgd1['q'][:,0],linewidth=2.5,color='orange',label='[-4,-2]')
    plt.plot(data_qsgd2['q'][:,0],linewidth=2.5,color='blue',label='[-2,0]')
    plt.plot(data_qsgd3['q'][:,0],linewidth=2.5,color='deeppink',label='[0,2]')
    plt.plot(data_qsgd4['q'][:,0],linewidth=2.5,color='green',label='[2,4]')
    plt.plot(data_qsgd5['q'][:,0],linewidth=2.5,color='red',label='[4,6]')
    plt.plot(data_qsgd6['q'][:,0],linewidth=2.5,color='black',label='[-6,6]')

    plt.xlabel('Epochs',fontsize=15)
    plt.ylabel('$z_x$',fontsize=15)
    plt.legend()

    plt.figure()
    plt.plot(data_qsgd1['z'][:,:,1],linewidth=0.5,color='yellow',alpha=0.1)
    plt.plot(data_qsgd2['z'][:,:,1],linewidth=0.5,color='cyan',alpha=0.1)
    plt.plot(data_qsgd3['z'][:,:,1],linewidth=0.5,color='deeppink',alpha=0.1)
    plt.plot(data_qsgd4['z'][:,:,1],linewidth=0.5,color='green',alpha=0.1)
    plt.plot(data_qsgd5['z'][:,:,1],linewidth=0.5,color='red',alpha=0.1)
    plt.plot(data_qsgd6['z'][:,:,1],linewidth=0.5,color='black',alpha=0.1)

    plt.plot(data_qsgd1['q'][:,1],linewidth=2.5,color='orange',label='[-4,-2]')
    plt.plot(data_qsgd2['q'][:,1],linewidth=2.5,color='blue',label='[-2,0]')
    plt.plot(data_qsgd3['q'][:,1],linewidth=2.5,color='deeppink',label='[0,2]')
    plt.plot(data_qsgd4['q'][:,1],linewidth=2.5,color='green',label='[2,4]')
    plt.plot(data_qsgd5['q'][:,1],linewidth=2.5,color='red',label='[4,6]')
    plt.plot(data_qsgd6['q'][:,1],linewidth=2.5,color='black',label='[-6,6]')

    plt.xlabel('Epochs',fontsize=15)
    plt.ylabel('$z_y$',fontsize=15)
    plt.legend()

if key=='proj':
    plt.figure()
    #plt.plot(data_qsgd1_proj1['cost'][-20:],label='1')
    plt.plot(data_qsgd1_proj2['cost'][-20:],label='2')
    plt.plot(data_qsgd1_proj5['cost'][-20:],label='5')
    plt.plot(data_qsgd1_proj10['cost'][-20:],label='10')
    plt.plot(data_qsgd1_proj100['cost'][-20:],label='100')
    plt.plot(data_qsgd4['cost'][-20:],color='black',ls='-.',label='Unprojected')

    plt.xlabel('Epochs',fontsize=15)
    plt.ylabel('$cost$',fontsize=15)
    plt.legend()

    plt.figure()
    plt.plot(data_qsgd1_proj1['q'][:,1],linewidth=2.5,color='orange',label='1')
    plt.plot(data_qsgd1_proj2['q'][:,1],linewidth=2.5,color='blue',label='2')
    plt.plot(data_qsgd1_proj5['q'][:,1],linewidth=2.5,color='deeppink',label='5')
    plt.plot(data_qsgd1_proj10['q'][:,1],linewidth=2.5,color='green',label='10')
    plt.plot(data_qsgd1_proj100['q'][:,1],linewidth=2.5,color='red',label='100')
    plt.plot(data_qsgd4['q'][:,1],linewidth=2.5,color='black',label='Unprojected',ls='-.')

    plt.xlabel('Epochs',fontsize=15)
    plt.ylabel('$z_x$',fontsize=15)
    plt.legend()

    plt.figure()
    plt.plot(data_qsgd1_proj1['q'][:,0],linewidth=2.5,color='orange',label='1')
    plt.plot(data_qsgd1_proj2['q'][:,0],linewidth=2.5,color='blue',label='2')
    plt.plot(data_qsgd1_proj5['q'][:,0],linewidth=2.5,color='deeppink',label='5')
    plt.plot(data_qsgd1_proj10['q'][:,0],linewidth=2.5,color='green',label='10')
    plt.plot(data_qsgd1_proj100['q'][:,0],linewidth=2.5,color='red',label='100')
    plt.plot(data_qsgd4['q'][:,0],linewidth=2.5,color='black',label='Unprojected',ls='-.')

    plt.xlabel('Epochs',fontsize=15)
    plt.ylabel('$z_y$',fontsize=15)
    plt.legend()

    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111)
    plt.grid(True)
    ax.add_patch(plt.Circle([0,0], radius=2.5, color='grey', alpha=0.3))
    ax.plot(data_qsgd4['z'][:,:,0],data_qsgd4['z'][:,:,1],color='red',linewidth=0.1,alpha=0.1)
    ax.plot(data_qsgd1_proj2['z'][:,:,0],data_qsgd1_proj2['z'][:,:,1],linewidth=0.1,color='cyan',alpha=0.1)
    # Start
    ax.add_patch(plt.Circle([1.75, 1.0], 0.1, color='g', alpha=0.3))
    # End
    ax.add_patch(plt.Circle([3.0,3.0], 0.1, color='r', alpha=0.3))
    ax.plot(X[:, 0], X[:, 1], 'r-', linewidth=2)

if key=='projection':
    plt.figure()
    plt.plot(data_qsgd1_proj['cost'],color='orange',label='Projected')
    plt.plot(data_qsgd3['cost'],color='deeppink',label='Unprojected')
    plt.legend()
    plt.figure()
    plt.plot(data_qsgd1_proj['q'][:,0],linewidth=2.5,color='orange',label='Projected')
    plt.plot(data_qsgd3['q'][:,0],linewidth=2.5,color='deeppink',label='Unprojected')
    plt.legend()
    plt.figure()
    plt.plot(data_qsgd1_proj['q'][:,0],linewidth=2.5,color='orange',label='Projected')
    plt.plot(data_qsgd3['q'][:,1],linewidth=2.5,color='deeppink',label='Unprojected')
    plt.legend()


if key=='example':
    obs = data_qsgd2['q'][-1,:]
    fig,ax = render_scene(obs,X)
