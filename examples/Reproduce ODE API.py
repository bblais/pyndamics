#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('pylab', 'inline')


# In[5]:


from scipy.integrate import odeint


# In[2]:


from pyndamics import Simulation
from pyndamics.emcee import *


# # A Differential Equation For Freefall
# 
# An object of mass $m$ is brought to some height and allowed to fall freely until it reaches the ground. A differential equation describing the object's speed over time is 
# 
# $$ y' = mg - \gamma y $$
# 
# The force the object experiences in the downwards direction is $mg$, while the force the object experiences in the opposite direction (due to air resistance) is proportional to how fast the object is presently moving. Let's assume the object starts from rest (that is, that the object's inital velocity is 0).  This may or may not be the case.  To showcase how to do inference on intial conditions, I will first assume the object starts from rest, and then relax that assumption later.
# 
# Data on this object's speed as a function of time is shown below.  The data may be noisy because of our measurement tools, or because the object is an irregular shape, thus leading to times during freefall when the object is more/less aerodynamic.  Let's use this data to estimate the proportionality constant for air resistance.
# 
# 

# In[6]:


# For reproducibility
np.random.seed(20394)

def freefall(y, t, p):    
    return 2.0*p[1] - p[0]*y[0]

# Times for observation
times = np.arange(0,10,0.5)
gamma,g, y0, sigma = 0.4, 9.8, -2, 2
y = odeint(freefall, t=times, y0=y0, args=tuple([[gamma,g]]))
yobs = np.random.normal(y,2)

fig, ax = plt.subplots(dpi=120)
plt.plot(times,yobs, label='observed speed', linestyle='dashed', marker='o', color='red')
plt.plot(times,y, label='True speed', color='k', alpha=0.5)
plt.legend()
plt.xlabel('Time (Seconds)')
plt.ylabel(r'$y(t)$');
plt.show()


# ```python
# ode_model = DifferentialEquation(
#     func=freefall,
#     times=times,
#     n_states=1, n_theta=2,
#     t0=0
# )
# 
# with pm.Model() as model:
#     # Specify prior distributions for soem of our model parameters
#     sigma = pm.HalfCauchy('sigma',1)    
#     gamma = pm.Lognormal('gamma',0,1)
# 
#     # If we know one of the parameter values, we can simply pass the value.
#     ode_solution = ode_model(y0=[0], theta=[gamma, 9.8])
#     # The ode_solution has a shape of (n_times, n_states)
# 
#     Y = pm.Normal('Y', mu=ode_solution, sd=sigma, observed=yobs)
# 
#     prior = pm.sample_prior_predictive()
#     trace = pm.sample(2000, tune=1000, cores=1)
#     posterior_predictive = pm.sample_posterior_predictive(trace)
# 
#     data = az.from_pymc3(trace=trace, prior=prior, posterior_predictive=posterior_predictive)
# ```

# In[25]:


sim=Simulation()
sim.add("y' = m*g - γ*y",-2)
sim.params(m=2,g=9.8,γ=0.4)
sim.add_data(t=times,y=yobs)
sim.run(0,10)


# In[26]:


t,y=sim.t,sim.y
plot(t,y)

t,y=sim.components[0].data['t'],sim.components[0].data['value']
plot(t,y,'o')


# In[27]:


model=MCMCModel(sim,
               _sigma=HalfCauchy(1),
                γ=LogNormal(0,1),
               )


# In[ ]:




