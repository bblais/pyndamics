# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <markdowncell>

# # Using Pyndamics to Perform Bayesian Parameter Estimation in Dynamical Systems
# 
# Pyndamics provides a way to describe a dynamical system in terms of the differential equations, or the stock-flow formalism. It is a wrapper around the Scipy odeint function, with further functionality for time plots, phase plots, and vector fields.  The MCMC component of this package uses emcee: http://dan.iel.fm/emcee/current/.  
# 
# Page for this package: [https://code.google.com/p/pyndamics/](https://code.google.com/p/pyndamics/)

# <codecell>

from pyndamics import Simulation
from pyndamics.emcee import *

# <markdowncell>

# ## Artificial Example with Mice Population

# <codecell>

data_t=[0,1,2,3]
data_mouse=[2,5,7,19]

sim=Simulation()                    # get a simulation object

sim.add("mice'=b*mice - d*mice",    # the equations
    2,                            # initial value
    plot=True)                      # display a plot, which is the default

sim.add_data(t=data_t,mice=data_mouse,plot=True)
sim.params(b=1.1,d=0.08)            # specify the parameters
sim.run(5)

# <codecell>

model=MCMCModel(sim,b=Uniform(0,10))

# <codecell>

model.set_initial_values()
model.plot_chains()

# <codecell>

model.run_mcmc(500)
model.plot_chains()

# <codecell>

model.run_mcmc(500)
model.plot_chains()

# <codecell>

model.best_estimates()

# <codecell>

sim.run(5)

# <codecell>

model.plot_distributions()

# <markdowncell>

# ## A linear growth example
# 
# Data from [http://www.seattlecentral.edu/qelp/sets/009/009.html](http://www.seattlecentral.edu/qelp/sets/009/009.html)

# <markdowncell>

# ### Plot the data

# <codecell>

t=array([7,14,21,28,35,42,49,56,63,70,77,84],float)
h=array([17.93,36.36,67.76,98.10,131,169.5,205.5,228.3,247.1,250.5,253.8,254.5])

plot(t,h,'-o')
xlabel('Days')
ylabel('Height [cm]')

# <markdowncell>

# ### Run an initial simulation
# 
# Here, the constant value ($a=1$) is hand-picked, and doesn't fit the data particularly well.

# <codecell>

sim=Simulation()
sim.add("h'=a",1,plot=True)
sim.add_data(t=t,h=h,plot=True)
sim.params(a=1)
sim.run(0,90)

# <markdowncell>

# ### Fit the model parameter, $a$
# 
# Specifying the prior probability distribution for $a$ as uniform between -10 and 10.

# <codecell>

model=MCMCModel(sim,a=Uniform(-10,10))

# <codecell>

model.run_mcmc(500)
model.plot_chains()

# <markdowncell>

# What is the best fit parameter value?

# <codecell>

model.best_estimates()

# <markdowncell>

# ### Rerun the model

# <codecell>

sim.run(0,90)

# <markdowncell>

# ### Plot the posterior histogram

# <codecell>

model.plot_distributions()

# <markdowncell>

# ### Fit the model parameter, $a$, and the initial value of the variable, $h$

# <codecell>

model=MCMCModel(sim,
                a=Uniform(-10,10),
                initial_h=Uniform(0,18),
                )

# <codecell>

model.run_mcmc(500)
model.plot_chains()

# <markdowncell>

# this looks like initial_h is irrelevant - or perhaps our uniform range is too small.

# <codecell>

model=MCMCModel(sim,
                a=Uniform(-10,10),
                initial_h=Uniform(0,180),
                )

# <codecell>

model.run_mcmc(500)
model.plot_chains()

# <codecell>

sim.run(0,90)
model.plot_distributions()

# <markdowncell>

# ### Plot the simulations for many samplings of the simulation parameters

# <codecell>

sim.noplots=True  # turn off the simulation plots
for i in range(500):
    model.draw()
    sim.run(0,90)
    plot(sim.t,sim.h,'g-',alpha=.05)
sim.noplots=False  # gotta love a double-negative
plot(t,h,'bo')  # plot the data

# <markdowncell>

# ## Logistic Model with the Same Data

# <codecell>

t=array([7,14,21,28,35,42,49,56,63,70,77,84],float)
h=array([17.93,36.36,67.76,98.10,131,169.5,205.5,228.3,247.1,250.5,253.8,254.5])

sim=Simulation()
sim.add("h'=a*h*(1-h/K)",1,plot=True)
sim.add_data(t=t,h=h,plot=True)
sim.params(a=1,K=500)
sim.run(0,90)

# fig=sim.figures[0]
# fig.savefig('sunflower_logistic1.pdf')
# fig.savefig('sunflower_logistic1.png')

# <markdowncell>

# ### Fit the model parameters, $a$ and $K$, and the initial value of the variable, $h$

# <codecell>

model=MCMCModel(sim,
                a=Uniform(.001,5),
                K=Uniform(100,500),
                initial_h=Uniform(0,100),
                )

# <markdowncell>

# when it looks weird, run mcmc again which continues from where it left off

# <codecell>

model.run_mcmc(500)
model.plot_chains()

# <codecell>

model.run_mcmc(500)
model.plot_chains()

# <codecell>

model.set_initial_values('samples')  # reset using the 16-84 percentile values from the samples
model.run_mcmc(500)
model.plot_chains()

# <codecell>

sim.a

# <codecell>

model.best_estimates()

# <markdowncell>

# ### Plot the Results

# <codecell>

sim.run(0,90)
model.plot_distributions()

# <markdowncell>

# ### Plot the joint distribution between parameters, $a$ and $K$

# <codecell>

model.triangle_plot()

# <markdowncell>

# ### Plot the many samples for predictions

# <codecell>

sim.noplots=True  # turn off the simulation plots
saved_h=[]
for i in range(500):
    model.draw()
    sim.run(0,150)
    plot(sim.t,sim.h,'g-',alpha=.05)
    saved_h.append(sim.h)
sim.noplots=False  # gotta love a double-negative
plot(t,h,'bo')  # plot the data
saved_h=array(saved_h)

# <codecell>

med=percentile(saved_h,50,axis=0)
lower=percentile(saved_h,2.5,axis=0)
upper=percentile(saved_h,97.5,axis=0)
plot(sim.t,med,'r-')
plot(sim.t,lower,'r:')
plot(sim.t,upper,'r:')

plot(t,h,'bo')  # plot the data

# <codecell>


