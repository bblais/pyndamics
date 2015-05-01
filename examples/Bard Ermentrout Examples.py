# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <markdowncell>

# ## ￼Simulating, Analyzing, and Animating Dynamical Systems by Bard Ermentrout

# <codecell>

from pyndamics import *

# <markdowncell>

# ## Section 2.1
# 
#     # Linear2d.ode
#     #
#     # right hand sides
#     x' =a*x+b*y
#     y'=c*x+d*y
#     #
#     # parameters
#     par a=0,b=l,c=-l,d=0
#     #
#     # some initial conditions init x=l,y=0
#     #
#     # we are done
#     done

# <codecell>

sim=Simulation()
sim.add("x'=a*x+b*y",1,plot=True)
sim.add("y'=c*x+d*y",0,plot=True)
sim.params(a=0,b=1,c=-1,d=0)
sim.run(0,20)

# <codecell>

phase_plot(sim,'x','y')
axis('equal')

# <markdowncell>

# ## Section 2.7
# 
#     # Fitzhugh-Nagumo equations
#     v'=I+v*(1-v)*(v-a) -w
#     w'=eps*(v-gamma*w)
#     par 1=0,a=.1,eps=.1,gamma=.25
#     @ xp=V,yp=w,xlo=-.25,xhi=l.25,ylo=-.5,yhi=l,total=100 @ maxstor=10000
#     done

# <codecell>

sim=Simulation()
sim.add("v'=I+v*(1-v)*(v-a) -w",0,plot=True)
sim.add("w'=eps*(v-gamma*w)",0,plot=True)
sim.params(I=0.4,a=0.1,eps=0.1,gamma=0.25)
sim.run(100)

# <codecell>

vector_field(sim,v=linspace(-.25,1.25,30),w=linspace(-.5,1,30))
phase_plot(sim,'v','w')

# <markdowncell>

# ## Section 3.2.1
# 
#     # forced duffing equation
#     x'=v
#     v'=a*x*(l-x~2) -f*v+c*cos(omega*t) 
#     par a=l,f=.2,c=.3,omega=l
#     init x=.l,v=.l 
#     done
# 

# <codecell>

sim=Simulation()
sim.add("x'=v",.1)
sim.add("v'=a*x*(1-x**2)-f*v+c*cos(omega*t)",.1)
sim.params(a=1,f=.2,c=.3,omega=1)
sim.run(200,num_iterations=20000)
phase_plot(sim,'x','v')

# <markdowncell>

# ## Section 3.3.1 - User defined functions
# 
#     # the wilson-cowan equations 
#     u'=-u+f(a*u-b*v+p) 
#     v'=-v+f(c*u-d*v+q) 
#     f(u)=!/(l+exp(-u)
#     par a=16,b=12,c=16,d=5,p=-l,q=-4
#     @ xp-u,yp=v,xlo=-.125,ylo=-.125,xhi=l,yhi=l 
#     done

# <codecell>

def f(u):
    return 1.0/(1+exp(u))

sim=Simulation()
sim.add("u'=-u+f(a*u-b*v+p)",0,plot=True)
sim.add("v'=-v+f(c*u-d*v+q)",0,plot=True)
sim.functions(f)
sim.params(a=16,b=12,c=16,d=5,p=-1,q=-4)
sim.run(10)

# <markdowncell>

# ## Section 3.4 - Auxiliary variables
# 
#     # the undamped pendulum 
#     theta'=v 
#     v'=-(g/1)*sin(theta) 
#     par g=9.8,1=2,ra=l
#     aux E=m*((l*v)~2/2+g*l*(1-cos(theta))) 
#     done

# <codecell>

sim=Simulation()
sim.add("theta'=v",10,plot=True)
sim.add("v'=-(g/L)*sin(theta)",plot=True)
sim.add("E=m*((L*v)**2/2+g*L*(1-cos(theta)))",plot=True)
sim.params(g=9.8,L=2,m=1)
sim.run(10)

# <markdowncell>

# ## Section 3.4.1 - Fixed Variables
# 
#     # standard nonlinear oscillator 
#     R=x"2+y~2
#     x'=x*(a-R)-y*(l+q*R) 
#     y'=y*(a-R)+x*(l+q*R)
#     par a=l,q=l 
#     done

# <codecell>

sim=Simulation()
sim.add("R=x**2+y**2")
sim.add("x'=x*(a-R)-y*(1+q*R)",10,plot=True)
sim.add("y'=y*(a-R)+x*(1+q*R)",0,plot=True)
sim.params(a=1,q=1 )
sim.run(10)

# <markdowncell>

# ## Section 4.5
# 
#     # example from Golubitsky #
#     x'=x+y+x~2-y~2+0.1 
#     y'=y-2*x*y+x~2/2+y~2
#     @ xp=x,yp=y
#     @ x l o = - 6 , x h i =6 , y l o = - 6 , y h i =6 , c l t = . 02 
#     done

# <codecell>

sim=Simulation()
sim.add("x'=x+y+x**2-y**2+0.1",0.1,plot=True)
sim.add("y'=y-2*x*y+x**2/2+y**2",0,plot=True)
sim.run(1.2)

# <codecell>

vector_field(sim,x=linspace(-6,6,30),y=linspace(-6,6,30))

# <codecell>


