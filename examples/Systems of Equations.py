# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

from pyndamics import Simulation

# <markdowncell>

# From http://wiki.scipy.org/Cookbook/CoupledSpringMassSystem

# <codecell>

sim=Simulation()
sim.add("x1'=y1",0.5,plot=1)
sim.add("y1'=(-b1 * y1 - k1 * (x1 - L1) + k2 * (x2 - x1 - L2)) / m1",0.0,plot=False)
sim.add("x2'=y2",2.25,plot=1)
sim.add("y2'=(-b2 * y2 - k2 * (x2 - x1 - L2)) / m2",0.0,plot=False)
sim.params( m1 = 1.0, # masses
            m2 = 1.5, # masses
            k1 = 8.0, # Spring constants
            k2 = 40.0,
            L1 = 0.5, # Natural lengths
            L2 = 1.0,
            b1 = 0.8, # Friction coefficients
            b2 = 0.5,
        )


sim.run(0,10) 

# <codecell>

sim=Simulation()
sim.add("x1''=(-b1 * x1' - k1 * (x1 - L1) + k2 * (x2 - x1 - L2)) / m1",[0.5,0.0],plot=[1,2])
sim.add("x2''=(-b2 * x2' - k2 * (x2 - x1 - L2)) / m2",[2.25,0.0],plot=[1,2])
sim.params( m1 = 1.0, # masses
            m2 = 1.5, # masses
            k1 = 8.0, # Spring constants
            k2 = 40.0,
            L1 = 0.5, # Natural lengths
            L2 = 1.0,
            b1 = 0.8, # Friction coefficients
            b2 = 0.5,
        )


sim.run(0,10) 

# <codecell>

system_equation="x[i]''=(-b[i] * x[i]' - k[i] * (x[i] -x[i-1] -L[i]) + k[i+1] * (x[i+1] - x[i] - L[i+1])) / m[i]"
system_params=dict(k=[8.0,40.0],
                   m=[1,1.5],
                   L=[0.5,1],
                   b=[0.8,0.5],
                   )

system_bc=dict(x_1=0,
               x_N=0,
               k_N=0,
               L_N=0,
               )


            
N=2

eqns=[]
for i in range(N):
    eqn=system_equation
    eqn=eqn.replace('[i]','%d' % i)
    if (i+1)<N:
        eqn=eqn.replace('[i+1]','%d' % (i+1))
    else:
        eqn=eqn.replace('[i+1]','_N')
    
    if (i-1)>=0:
        eqn=eqn.replace('[i-1]','%d' % (i-1))
    else:
        eqn=eqn.replace('[i-1]','_%d' % abs((i-1)))
        
    eqns.append(eqn)
    
print system
for eqn in eqns:
    print eqn
    
print
print "x1''=(-b1 * x1' - k1 * (x1 - L1) + k2 * (x2 - x1 - L2)) / m1".replace('1','0').replace('2','1')
print "x2''=(-b2 * x2' - k2 * (x2 - x1 - L2)) / m2".replace('1','0').replace('2','1')
    

# <codecell>

def add_system(sim,N,equations,initial_values,plot=True,**kwargs):
    if isinstance(equations,str):
        equations=[equations]
        
    plot_values=plot
    
    for system_equation in equations:
        for i in range(N):
            val=initial_values[i]
            
            try:
                plot=plot_values[i]
            except TypeError:
                plot=plot_values
            
            eqn=system_equation
            eqn=eqn.replace('[i]','%d' % i)
            if (i+1)<N:
                eqn=eqn.replace('[i+1]','%d' % (i+1))
            else:
                eqn=eqn.replace('[i+1]','_N')
            
            if (i-1)>=0:
                eqn=eqn.replace('[i-1]','%d' % (i-1))
            else:
                eqn=eqn.replace('[i-1]','_%d' % abs((i-1)))
                
            sim.add(eqn,val,plot=plot)

def add_system_params(sim,N,**kwargs):
    params={}
    for name in kwargs:
        if name.endswith('_1') or name.endswith('_N'):
            params[name]=kwargs[name]
        else:
            for i in range(N):
                newname=name+"%d" % i
                try:
                    params[newname]=kwargs[name][i]
                except TypeError:
                    params[newname]=kwargs[name]
            
    sim.params(**params)
    
    
    
N=2
sim=Simulation()
add_system(sim,N,"x[i]''=(-b[i] * x[i]' - k[i] * (x[i] -x[i-1] -L[i]) + k[i+1] * (x[i+1] - x[i] - L[i+1])) / m[i]",
           [[0.5,0.0],[2.25,0.0]],  # initial conditions
           [[1,2],[1,2]],  # plot arguments
           )

add_system_params(sim,N,
            k=[8.0,40.0],
            m=[1,1.5],
            L=[0.5,1],
            b=[0.8,0.5],
            
            x_1=0,
            x_N=0,
            k_N=0,
            L_N=0,
        )

print sim.equations()
sim.run(0,10) 

# <codecell>

N=3
sim=Simulation()
add_system(sim,N,"x[i]''=(-b[i] * x[i]' - k[i] * (x[i] -x[i-1] -L[i]) + k[i+1] * (x[i+1] - x[i] - L[i+1])) / m[i]",
           [[0.5,0.0],[2.25,0.0],[4.0,0.0]],  # initial conditions
           [[1,2],[1,2],[1,2]],  # plot arguments
           )

add_system_params(sim,N,
            k=[8.0,40.0,10.0],
            m=[1,1.5,1.0],
            L=[0.5,1,2.0],
            b=[0.8,0.5,.7],
            
            x_1=0,
            x_N=0,
            k_N=0,
            L_N=0,
        )

print sim.equations()
sim.run(0,10) 

# <codecell>

A=zeros((3,3))

b=[0.8,0.5]
L=[0.5,1]
m=[1,1.5]
k=[8.0,40.0]




# <codecell>

system_equation="x[i]''=(-b[i] * x[i]' - k[i] * (x[i] -x[i-1] -L[i]) + k[i+1] * (x[i+1] - x[i] - L[i+1])) / m[i]"
system_params=dict(k=[8.0,40.0],
                   m=[1,1.5],
                   L=[0.5,1],
                   b=[0.8,0.5],
                   )

system_bc=dict(x_1=0,
               x_N=0,
               k_N=0,
               L_N=0,
               )


            
N=2

eqns=[]
for i in range(N):
    eqn=system_equation
    eqn=eqn.replace('[i]','%d' % i)
    if (i+1)<N:
        eqn=eqn.replace('[i+1]','%d' % (i+1))
    else:
        eqn=eqn.replace('[i+1]','_N')
    
    if (i-1)>=0:
        eqn=eqn.replace('[i-1]','%d' % (i-1))
    else:
        eqn=eqn.replace('[i-1]','_%d' % abs((i-1)))
        
    eqns.append(eqn)
    
print system
for eqn in eqns:
    print eqn
    
print
print "x1''=(-b1 * x1' - k1 * (x1 - L1) + k2 * (x2 - x1 - L2)) / m1".replace('1','0').replace('2','1')
print "x2''=(-b2 * x2' - k2 * (x2 - x1 - L2)) / m2".replace('1','0').replace('2','1')
    

