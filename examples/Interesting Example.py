#!/usr/bin/env python
# coding: utf-8

# http://tutorial.math.lamar.edu/Classes/DE/EulersMethod.aspx

# In[1]:


get_ipython().run_line_magic('pylab', 'inline')


# In[2]:


from pyndamics import Simulation


# In[4]:


sim=Simulation()
sim.add("y'=y - 1/2*exp(t/2)*sin(5*t) + 5*exp(t/2)*cos(5*t)",0)
sim.run(0,5)


# In[6]:


t,y=sim.t,sim.y
plot(t,y)
y2=exp(t/2)*sin(5*t)
plot(t,y2,'r--')


# In[ ]:




