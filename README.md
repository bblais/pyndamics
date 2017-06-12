This package provides a way to describe a dynamical system in terms of the differential equations, or the stock-flow formalism.  It is a wrapper around the Scipy odeint function, with further functionality for time plots, phase plots, and vector fields.

See http://nbviewer.ipython.org/7321928 for many examples.

A simple example is the (Lorenz System)[http://en.wikipedia.org/wiki/Lorenz_system]:

![Lorenz System formulas](http://upload.wikimedia.org/math/b/0/e/b0ea9119f6aaa31302164c4212cce72d.png)

implemented in pyndamics as

    from pyndamics import Simulation

    sim=Simulation()
    sim.add("x'=sigma*(y-x)",14,plot=True)
    sim.add("y'=x*(rho-z)-y",8.1,plot=True)
    sim.add("z'=x*y-beta*z",45,plot=True)
    sim.params(sigma=10,beta=8.0/3,rho=15)
    sim.run(0,50)  

The package also allows for statistical inference on the parameters, using the Markov Chain Monte Carlo (MCMC) technique implemented in PyMC version 2 (https://github.com/pymc-devs/pymc/tree/2.3) or the MCMC Hammer (http://dan.iel.fm/emcee/current/)

See http://nbviewer.ipython.org/gist/bblais/7807979 for many MCMC examples using Pyndamics.
