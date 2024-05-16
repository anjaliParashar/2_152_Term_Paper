import jax 
from jax import random
import jax.numpy as jnp


def ULA(Zi, gradient, seed, step=0.001):
    key = random.PRNGKey(seed) 
    n = Zi.shape[1]
    Zi = Zi - step * gradient + jnp.sqrt(2 * step) * random.uniform(key, shape=(1,n), minval=-1.5, maxval=1.5)
    return Zi

def QSGD(Zi,gradient,seed,K,p,step=0.01):
    key = random.PRNGKey(int(seed))
    l=2.5
    ord = (l-2)/(l-1)
    Z_master = jnp.sum(Zi,axis=0)/p
    Z_master = jnp.repeat(Z_master[jnp.newaxis, :, :], Zi.shape[0], axis=0)
    #print((Z_master-Zi).shape,jnp.diag(K).shape)
    Zi = Zi - step * gradient + jnp.tensordot(K,(Z_master-Zi),(0,0)) -step* random.uniform(key, shape=(Zi.shape[0],Zi.shape[1],Zi.shape[2]), minval=-1.5, maxval=1.5)
    return Zi

def SGD(Zi,gradient,seed,k,p,step=0.01):
    key = random.PRNGKey(int(seed))
    l=2.5
    ord = (l-2)/(l-1)
    Z_master = jnp.sum(Zi,axis=0)/p
    Zi = Zi - step * gradient  -step* random.uniform(key, shape=(Zi.shape[0],Zi.shape[1]), minval=-1.5, maxval=1.5)
    return Zi
