
# coding: utf-8

# In[1]:

from pyndamics import Simulation
from pyndamics.emcee import *


# In[2]:

sim=Simulation()
sim.add("y'=a*y*(1-y/K)",.1,plot=True)
sim.params(a=1,K=10)
sim.run(0,10)


# In[3]:

model=MCMCModel(sim,a=Uniform(.001,5),K=Uniform(6,20),initial_h=Normal(4))


# In[ ]:



