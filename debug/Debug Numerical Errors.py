
# coding: utf-8

# In[1]:

get_ipython().magic('pylab inline')


# In[2]:

from pyndamics import Simulation
from pyndamics.emcee import *


# In[6]:

t=linspace(0,10,100)
h=3.4*exp(.3*t)

sim=Simulation()
sim.add("h'=a*h",1,plot=True)
sim.add_data(t=t,h=h,plot=True)
sim.params(a=1)
sim.run(0,10)


# In[14]:

model=MCMCModel(sim,
                a=Normal(0,10),
                initial_h=Uniform(0,100),
                )


# In[17]:

model.run_mcmc(500)
model.set_initial_values('samples')


# In[18]:

model.plot_distributions()


# In[ ]:



