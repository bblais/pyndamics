from __future__ import print_function


from scipy.integrate import odeint,ode
from pylab import *
from numpy import *
import pylab
from copy import deepcopy 

from matplotlib import rc
size=20
family='sans-serif'

rc('font',size=size,family=family)
rc('axes',titlesize=size,grid=True,labelsize=size)
rc('xtick',labelsize=size)
rc('ytick',labelsize=size)
rc('legend',fontsize=size)
rc('lines',linewidth=2)
rc('figure',figsize=(12,8))


import os
import sys

class RedirectStdStreams(object):
    def __init__(self, stdout=None, stderr=None):
        self._stdout = stdout or sys.stdout
        self._stderr = stderr or sys.stderr

    def __enter__(self):
        self.old_stdout, self.old_stderr = sys.stdout, sys.stderr
        self.old_stdout.flush(); self.old_stderr.flush()
        sys.stdout, sys.stderr = self._stdout, self._stderr

    def __exit__(self, exc_type, exc_value, traceback):
        self._stdout.flush(); self._stderr.flush()
        sys.stdout = self.old_stdout
        sys.stderr = self.old_stderr

devnull = open(os.devnull, 'w')


def from_values(var,*args):
    if len(args)==1:        
        y=[v[1] for v in args[0]]
        x=[v[0] for v in args[0]]
    else:
        y=[v[1] for v in args]
        x=[v[0] for v in args]
        
    if var<x[0]:
        return y[0]
    if var>x[-1]:
        return y[-1]
    
    return interp(var,x,y)


def array_wrap(f):
    # allow a function to be written for float values, but be handed array
    # values and return an array
    
    def what(*args,**kw):
        try:
            val=f(*args,**kw)
        except ValueError:  # array treated as float
            for a in args:
                if isinstance(a,ndarray):
                    L=len(a)
                    break
            val=[]
            for i in range(L):
                newargs=[]
                for a in args:
                    if isinstance(a,ndarray):
                        newargs.append(a[i])
                    else:
                        newargs.append(a)
                newargs=tuple(newargs)
                
                val.append(f(*newargs,**kw))
            val=array(val)
#        except TypeError: # float treated as array
        
        return val
        
    return what

def mapsolve(function,y0,t_mat,*args):

    y=array(y0)

    y_mat=[]
    for t in t_mat:
        y_mat.append(y)
        newy=array(function(y,t,*args))
        y=newy

    ret=array(y_mat)

    return ret


def euler(function,y0,t_mat,*args,**kwargs):

    dt=t_mat.ravel()[1]-t_mat.ravel()[0]

    y=array(y0)

    y_mat=[]
    for t in t_mat:
        y_mat.append(y)
        dy=array(function(y,t,*args))*dt  # call the function
        y=y+dy
        


    ret=array(y_mat)
    
    
    return ret

def rk2(function,y0,t_mat,*args,**kwargs):

    dt=t_mat.ravel()[1]-t_mat.ravel()[0]

    y=array(y0)

    y_mat=[]
    for t in t_mat:
        y_mat.append(y)

        tp=t+dt/2.0        
        dyp=array(function(y,t,*args))*dt  # call the function
        yp=y+0.5*dyp
        
        dy=array(function(yp,tp,*args))*dt
        y=y+dy


    ret=array(y_mat)
    
    return ret

def rk4(function,y0,t_mat,*args,**kwargs):

    dt=t_mat.ravel()[1]-t_mat.ravel()[0]

    y=array(y0)

    y_mat=[]
    for t in t_mat:
        y_mat.append(y)

        y1=y
        t1=t
        f1=array(function(y1,t1,*args))
        
        y2=y+0.5*f1*dt
        t2=t+0.5*dt
        f2=array(function(y2,t2,*args))
        
        y3=y+0.5*f2*dt
        t3=t+0.5*dt
        f3=array(function(y3,t3,*args))
        
        y4=y+f3*dt
        t4=t+dt
        f4=array(function(y4,t4,*args))

        dy=1.0/6.0*(f1+2*f2+2*f3+f4)*dt
        y=y+dy


    ret=array(y_mat)
    
    return ret


def rkwrapper(function,_self):
    def fun(t,y,*args,**kwargs):
        return function(y,t,_self,*args,**kwargs)
    return fun

def rk45(function,y0,t_mat,_self,*args,**kwargs):

    dt=t_mat.ravel()[1]-t_mat.ravel()[0]
    t0=t_mat.ravel()[0]
    t1=t_mat.ravel()[-1]
    function=rkwrapper(function,_self)
    r=ode(function).set_integrator('dopri5',nsteps=300)
    r.set_initial_value(y0, t0)
    
    y=array(y0)

    y_mat=[]
    y_mat.append(r.y)
    while r.successful() and r.t <= t1:
        r.integrate(r.t+dt)
        y_mat.append(r.y)


    ret=array(y_mat)

    return ret


def simfunc(_vec,t,_sim):
    
    if _sim.method=='map':
        usemap=True
    else:
        usemap=False
        
    _l=locals()
    for _i,_c in enumerate(_sim.components):
        _l[_c.name]=_vec[_i]
    
    _l.update(_sim.myparams)
    
    for _i,_c in enumerate(_sim.assignments):
        _s='%s' % _c.diffstr
        if _sim.verbose:
            print(_s)
            
        try:
            _val=eval(_s,_l)
        except NameError:
            for _j in range(_i+1):
                print(_sim.assignments[_j].diffstr)
            
            _val=eval(_s,_l)
        _l[_c.name]=_val

    
    _diff=[]
    for _i,_c in enumerate(_sim.components):
        if not _c.diffstr:
            _s='0' 
        else:
            _s='%s' % _c.diffstr
        
        if _sim.verbose:
            print(_s)
            
        _val=eval(_s,_l)
        
        if not _c.min is None:
            if _vec[_i]<_c.min:
                _vec[_i]=_c.min
                if _val<0:  # stop the change in the variable
                    _val=0
                
        if not _c.max is None:
            if _vec[_i]>_c.max:
                _vec[_i]=_c.max
                if _val>0: # stop the change in the variable
                    _val=0
    
        _diff.append(_val)
        
        if usemap:
            _l[_c.name]=_val

    return _diff

def phase_plot(sim,x,y,z=None,**kwargs):
    """
    Make a Phase Plot of two or three variables.

    Parameters
    ----------
    sim : Simulation
        This is a simulation object.
    x : str
        Name of the variable to plot on the x-axis
    y : str
        Name of the variable to plot on the y-axis
    z : str, optional
        Name of the variable to plot on the (optional) z-axis
 
    Returns
    -------
    """

    from mpl_toolkits.mplot3d import Axes3D
    if not z is None:  # 3D!
        ax = gcf().add_subplot(111, projection='3d')
        ax.plot(sim[x],sim[y],sim[z])
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.set_zlabel(z)    
    else:
        plot(sim[x],sim[y])
        xlabel(x)
        ylabel(y)
        

def vector_field(sim,rescale=False,**kwargs):
    keys=sorted(kwargs.keys())
    
    tuples=[ kwargs[key] for key in keys ]
    if len(tuples)==1:
        tuples.append(array([0]))
    
    X,Y=meshgrid(*tuples)
    
    U=zeros_like(X)
    V=zeros_like(X)

    count=0
    for x,y in zip(X.ravel(),Y.ravel()):
            
        vec=[]
        for i,c in enumerate(sim.components):
            if i==0: # set x
                c.initial_value=x
            elif i==1: # set y
                c.initial_value=y
            else:
                raise ValueError("Not Implemented for 3D+")
                
            vec.append(c.initial_value)
        vec=array(vec)
        t=0
        df=simfunc(vec,t,sim)
        
        U.ravel()[count]=df[0]
        try:
            V.ravel()[count]=df[1]
        except IndexError:
            pass
        
        count+=1

    if rescale:
        N = sqrt(U**2+V**2)  # there may be a faster numpy "normalize" function
        U, V = U/N, V/N
        
    figure(figsize=sim.figsize)
    Q = quiver(  X, Y, U, V)
    xlabel(sim.components[0].name)
    try:
        ylabel(sim.components[1].name)
    except IndexError:
        pass

    



class Component(object):
    
    def __init__(self,diffstr,initial_value=0,
                        min=None,max=None,
                        plot=False,save=None):

        name,rest=diffstr.split('=')
        name=name.strip()
        self.orig_diffstr=diffstr
        
        if "'" in name:
            name=name[:-1]
            self.diffeq=True
        else:
            self.diffeq=False
        
        self.diffstr=rest.strip()

        self.diffstr=self.replace_primes(self.diffstr)

        self.name=name
        self.initial_value=initial_value
        
        try:
            self.length=len(initial_value)
            if self.length==1:
                self.initial_value=initial_value[0]
        except TypeError:
            self.length=1
            
        self.values=None
        self.min=min
        self.max=max
        self.plot=plot
        self.save=save
        
        if name.endswith("_"):  # a primed name
            ps=name.split('_')
            self.label="_".join(ps[:-2])+"'"*len(ps[-2])
        else:
            self.label=name
            
        self.data={}
    
    def inflow(self,s):
        
        s=s.strip()
        s=self.replace_primes(s)
        
        self.diffstr+='+'+s

    def outflow(self,s):

        s=s.strip()
        s=self.replace_primes(s)

        self.diffstr+='-('+s+')'

    def replace_primes(self,s):
        import re
        
        s2=s
        
        if "'" not in s2:
            return s2
            
        for i in range(10,0,-1):
            s2=re.sub("(\w*)"+"'"*i,"\\1_"+"p"*i+"_",s2)
        
        
        return s2
        
    def __getitem__(self,s):
        """docstring for __getitem__"""
        return self.values[s]
        
    def __repr__(self):
        s="%s : %s\n%s" % (self.label,self.orig_diffstr,str(self.values))
        return s
        
        
numpy_functions=(sin,cos,exp,tan,abs,floor,ceil,radians,degrees,
                         sinh,cosh,tanh,arccos,arcsin,arctan,arctan2,
                         min,max,sqrt,log,log10,mean,median)
class Simulation(object):
    
    def __init__(self,method='odeint',verbose=False,plot_style='.-'):
        
        self.initialized=False
        self.components=[]
        self.assignments=[]
        self.plot_style=plot_style
        self.use_delays=False
        self.data_delay={}
        
        self.verbose=verbose
        self.myparams={}
        self.method=method
        self.show=True
        self.original_params={}
        self.initial_value={}
        self.extra={}
        self.omit=[]
        
        self.figsize=(12,8)
        self.myparams.update({'from_values':array_wrap(from_values)})
        self.functions(*numpy_functions,omit=True)
        self.myparams.update(pi=pi,inf=inf)
        
        self.omit.append('pi')
        self.omit.append('inf')
        
        self.use_func=True
        self.func=None
        self.myfunctions={}
        self.initialized=True
        self.maximum_data_t=-1e500
        self.maximum_t=-1e500
        self.noplots=False
        
        self.figures=[]
        
    def delay(self,var,t):
        return interp(t,self.data_delay[var]['t'],self.data_delay[var]['value'])
        
    def make_func(self):
        from numba import jit
    
        all_eq=True
        all_diffeq=True
        for c in self.components:
            if c.diffeq:
                all_eq=False
            if not c.diffeq:
                all_diffeq=False
                

        if not all_eq and not all_diffeq:
            components=[]
            for c in self.components:
                if not c.diffeq:
                    self.assignments.append(c)
                else:
                    components.append(c)
            self.components=components
            
    
        _sim=self

        s="def _simfunc(_vec,t,_sim):\n"

        for _f in self.myfunctions:
            s=s+"    %s=_sim.myfunctions['%s']\n" % (_f,_f)
        
        for _i,_c in enumerate(_sim.components):
            s=s+"    initial_%s=_sim.initial_value['%s']\n" % (_c.name,_c.name)
        s=s+"\n"
        
        for key in _sim.original_params:
            s=s+"    %s=_sim.original_params['%s']\n" % (key,key)
        s=s+"\n"

        for _i,_c in enumerate(_sim.components):
            s=s+"    %s=_vec[%d]\n" % (_c.name,_i)
        s=s+"\n"



        if _sim.use_delays:
            for _i,_c in enumerate(_sim.components):
                s+="    _sim.data_delay['%s']['t'].append(t)\n" % (_c.name)
                s+="    _sim.data_delay['%s']['value'].append(%s)\n" % (_c.name,_c.name)

        for _i,_c in enumerate(_sim.assignments):
            s=s+"    %s=%s\n" % (_c.name,_c.diffstr)
            if _sim.use_delays:
                s+="    _sim.data_delay['%s']['t'].append(t)\n" % (_c.name)
                s+="    _sim.data_delay['%s']['value'].append(%s)\n" % (_c.name,_c.name)
            
            
        s=s+"\n"


        s=s+"    _diff=[]\n"
        for _i,_c in enumerate(_sim.components):
            s=s+"    _val=%s\n" % (_c.diffstr)
            if not _c.min is None:            
                s=s+"""
    if _vec[%d]<%s:
        _vec[%d]=%s
        if _val<0:
            _val=0
                """ % (_i,_c.min,_i,_c.min)
                s=s+"\n"

            if not _c.max is None:            
                s=s+"""
    if _vec[%d]>%s:
        _vec[%d]=%s
        if _val>0:
            _val=0
                """ % (_i,_c.max,_i,_c.max)
                s=s+"\n"

            s=s+"    _diff.append(_val)\n"
            
        s=s+"\n"

        s=s+"    return _diff\n"

        self.func_str=s
        exec(s)

        _sim.func=locals()['_simfunc']

        return locals()['_simfunc']
         
        
    def inflow(self,cname,s):
        
        c=[x for x in self.components if x.name==cname]
        
        if not c:
            raise ValueError('No component named "%s"' % cname)
            
        c[0].inflow(s)
        
    def outflow(self,cname,s):

        c=[x for x in self.components if x.name==cname]

        if not c:
            raise ValueError('No component named "%s"' % cname)

        c[0].outflow(s)

    def stock(self,name,initial_value=0,
                    min=None,max=None,
                    plot=False,save=None):
                    
        c=Component(name+"'=",initial_value,min,max,plot,save)
        self.components.append(c)
        return c
        
        
    def equations(self):
        s=""
        for c in self.components:
            if c.diffstr:
                s+="%s'=%s\n" % (c.name,c.diffstr)
            else:
                s+="%s'=0\n" % (c.name)
            
        for key in self.myparams:
            if not key in ['from_values']:
                if key in self.omit:
                    continue
            
                s+="%s=%s\n" % (key,str(self.myparams[key]))
            
        return s
            
    def copy(self):
    
        sim_copy=Simulation()
        for c in self.components:
            
            c_copy=Component(c.label+"'="+c.diffstr,c.initial_value,
                        c.min,c.max,c.plot,c.save)
            sim_copy.components.append(c_copy)
            
        for key in self.myparams:
            sim_copy.myparams[key]=self.myparams[key]
            
        sim_copy.method=self.method
        sim_copy.verbose=self.verbose
        
        return sim_copy
        
    def add(self,diffstr,initial_value=0,
                min=None,max=None,
                plot=False,save=None):
                
        name,rest=diffstr.split('=')
        
        if 'delay(' in rest or 'delay (' in rest:
            self.use_delays=True
            
        if name.count("'")<=1:
            c=Component(diffstr,initial_value,min,max,plot,save)
            self.components.append(c)
            self.data_delay[c.name]={'t':[],'value':[]}
            return c
        else:  # higher order of diffeq
            order=name.count("'")
            name=name.split("'")[0]
            cc=[]
            # make the new variables
            
            try:
                L=len(plot)
                assert L==order
            except TypeError:
                plot=[plot]*order
            
            
            for i in range(order-1):
                if i==0:
                    vname1=name
                else:
                    vname1=name+"_"+"p"*i+"_"
                vname2=name+"_"+"p"*(i+1)+"_"
                
                ds="%s'=%s" % (vname1,vname2)
                
                c=Component(ds,initial_value[i],min,max,plot[i],save)
                
                self.components.append(c)
                cc.append(c)
                
            vname=name+"_"+"p"*(order-1)+"_"
            ds="%s'=%s" % (vname,rest)
            c=Component(ds,initial_value[order-1],min,max,plot[-1],save)
            self.components.append(c)
            self.data_delay[c.name]={'t':[],'value':[]}
            cc.append(c)
            
        
    def initial_values(self,**kwargs):
            self.initial_value.update(kwargs)
        
    def params(self,**kwargs):
        for name in kwargs:
            if name in [c.name for c in self.components]:
                raise ValueError("Parameter name %s already a variable." % name)


        self.myparams.update(kwargs)
        self.original_params.update(kwargs)
        
    def functions(self,*args,**kwargs):
        
        try:
            omit=kwargs['omit']
        except KeyError:
            omit=False
            
        #self.myparams.update(dict(zip([f.__name__ for f in args],args)))
        
        self.myparams.update(dict(list(zip(
            [f.__name__ for f in args],
            [array_wrap(a) for a in args],
            ))))        
        
        if self.initialized:
            for key in [f.__name__ for f in args]:
                self.myfunctions[key]=self.myparams[key]
             
        if omit:
            self.omit.extend([f.__name__ for f in args])   
                
    def vec2str(self,vec):
        
        s=""
        i=0
        for c in self.components:
            s+="%s=vec[%d]\n" % (c.name,i)
            i+=1
            
        return s
            
    def post_process(_sim):

        t=_sim.t
        
        try:  # assignments with arrays
            _l=locals()
            for _i,_c in enumerate(_sim.components):
                _l[_c.name]=_c.values
                _l["initial_"+_c.name]=_c.initial_value

            _l.update(_sim.myparams)

            for _c in _sim.assignments:
                _s='%s' % _c.diffstr
                if _sim.verbose:
                    print(_s)
                
                try:
                    _val=eval(_s)
                except FloatingPointError:
                    if 'post_process error' not in _sim.extra:
                        _sim.extra['post_process error']=[]
                        
                    _sim.extra['post_process error'].append("Floating Point Error on '%s'" % (_s))
                    _val=-1e500
                                    
                _l[_c.name]=_val
                _c.values=_val
        except ValueError:
            _l=locals()
            _c=_sim.components[0]
            _N=len(_c.values)
            
            for _c in _sim.assignments:
                _c.values=[]
                
            for _j in range(_N):
                for _i,_c in enumerate(_sim.components):
                    _l[_c.name]=_c.values[_j]

                _l.update(_sim.myparams)

                for _c in _sim.assignments:
                    _s='%s' % _c.diffstr
                    if _sim.verbose:
                        print(_s)
                    _val=eval(_s)                        
                    _l[_c.name]=_val
                    _c.values.append(_val)
            
            for _c in _sim.assignments:
                _c.values=array(_c.values,float)
            

    def mse(self,name):
        svals=self[name]
        st=self['t']

        c=self.get_component(name)

        if not c.data:
            raise ValueError('No Data for MSE')

        simvals=interp(c.data['t'],self.t,c.values)
        mseval=((array(c.data['value'])-simvals)**2).mean()

        return mseval

    def compare(self,cname,t=None,value=None,
            plot=False,transform=None):
        if t is None or value is None:
            return None
            
        svals=self[cname]
        st=self['t']
        
        if transform=='log':
            svals=log(svals)
            value=log(array(value))
            label='log(%s)' % cname
        else:
            label=cname

        simvals=interp(t,st,svals)
        
        mse=((array(value)-simvals)**2).mean()
        
        if plot:
            fig=figure(figsize=self.figsize)
            self.figures.append(fig)
            
            pylab.plot(st,svals,'-o')
            pylab.plot(t,value,'rs')
            pylab.grid(True)
            xlabel('time')
            ylabel(label)
            title('MSE : %.3e' % mse)
            draw()
            if self.show:
                show()
        
        
        return mse        

    def repeat(self,t_min,t_max=None,num_iterations=1000,**kwargs):
        keys=list(kwargs.keys())
        name=keys[0]
        L=len(kwargs[name])
        results=[]
        
        self.noplots=True
        for i in range(L):
            result={}
            
            params={}
            for name in keys:
                params[name]=kwargs[name][i]
                
            self.params(**params)
            self.run_fast(t_min,t_max,num_iterations)
            
            for c in self.components:
                result[c.name]=c.values
        
            results.append(result)
            
        self.noplots=False
        return results

    def interpolate(self,t,cname=None):
        if cname is None:
            cnames=[x.name for x in self.components]
            result={}
            for cname in cnames:
                result[cname]=self.interpolate(t,cname)
            return result
        else:
    
            svals=self[cname]
            st=self['t']
            
            # gives a value error if svals is nan
            try:
                simvals=interp(t,st,svals)
            except ValueError:
                try:
                    simvals=-1e500*ones(len(t))
                except TypeError:  # a float given
                    simvals=-1e500
                
            return simvals
       
    def run_fast(self,t_min=None,t_max=None,num_iterations=1000,**kwargs):
        if t_min is None:
            assert self.maximum_data_t>-1e500
            
            t_min=0
            t_max=self.maximum_data_t+0.1

        if not self.func:
            self.make_func()
            
        t=linspace(t_min,t_max,num_iterations)
        y0=[c.initial_value for c in self.components]
        for c in self.components:
            self.initial_value[c.name]=c.initial_value

        func=self.func

        # I got sick of the ridiculous messages when trying to run fast
        with RedirectStdStreams(stdout=devnull, stderr=devnull):
            result,extra=odeint(func,y0,t,(self,),full_output=True,printmessg=False,**kwargs)
        
        self.extra=extra
        self.t=t
        for i in range(result.shape[1]):
            self.components[i].values=result[:,i]

        if self.assignments:
            self.post_process()
        
    def run(self,t_min=None,t_max=None,num_iterations=1000,dt=None,discrete=False,xlabel='Time',**kwargs):
        self.figures=[]
    
        if t_min is None:
            assert self.maximum_data_t>-1e500
            
            t_min=0
            t_max=self.maximum_data_t+0.1
    
    
        if t_max is None:
            try:
                t_array=t_min
                t_max=max(t_array)
                t_min=min(t_array)
            except TypeError:  # an int or a float
                t_max=t_min
                t_min=0

        all_eq=True
        all_diffeq=True
        for c in self.components:
            if c.diffeq:
                all_eq=False
            if not c.diffeq:
                all_diffeq=False
                

        if not all_eq and not all_diffeq:
            components=[]
            for c in self.components:
                if not c.diffeq:
                    self.assignments.append(c)
                else:
                    components.append(c)
            self.components=components
            
        
        if all_eq:
            self.method='map'
        
        if self.method=='map':
            discrete=True
            
            if all_diffeq:
                raise ValueError("Cannot have map and diffeq.")
                
        
        if discrete:
            t_min=int(t_min)
            t_max=int(t_max)
            num_iterations=t_max-t_min+1
        
        vec=[]
        for c in self.components:
            vec.append(c.initial_value)
            self.initial_value[c.name]=c.initial_value

        vec=array(vec)
            
        t=t_min
        self.maximum_t=t_max
        
        if not self.method=='odeint':
            #assert not self.use_func
            self.use_func=False
        
        
        if self.use_func and self.func is None:
            self.make_func()
    
        if self.use_func:
            func=self.func
        else:
            func=simfunc
        
        
        df=func(vec,t,self)

        if not dt is None:
            num_iterations=int((t_max-t_min)/dt)

        t=linspace(t_min,t_max,num_iterations)

        y0=[c.initial_value for c in self.components]
        
        
        if self.method=='map':
            result=mapsolve(func,y0,t,self)
        elif self.method=='euler' or discrete:
            result=euler(func,y0,t,self)
        elif self.method=='odeint':
            result,extra=odeint(func,y0,t,(self,),full_output=True,printmessg=False,**kwargs)
            self.extra=extra
        elif self.method=='rk2':
            result=rk2(func,y0,t,self,**kwargs)
        elif self.method=='rk4':
            result=rk4(func,y0,t,self,**kwargs)
        elif self.method=='rk45':
            result=rk45(func,y0,t,self,**kwargs)
        else:
            raise TypeError("Unknown method: '%s'" % (self.method))
        

        self.t=t
        for i in range(result.shape[1]):
            self.components[i].values=result[:len(t),i]

        if self.assignments:
            self.post_process()

        count=1
        legends={1:[]}
        self.max_figure=0
        drawit=False  # flag to call draw()
        for c in self.components+self.assignments:
            
            if not c.save is None:
                with open(c.save,'wt') as fid: 
                    for tt,vv in zip(t,c.values):
                        fid.write('%f,%f\n' % (tt,vv))
            
            
            if c.plot and not self.noplots:
                
                if c.plot is True:
                    fig=figure(count,figsize=self.figsize)
                    if fig not in self.figures:
                        self.figures.append(fig)
                        
                    self.max_figure=count
                    clf()
                    
                    legends[count].append(c.label)
                    count+=1
                    legends[count]=[]
                else:
                    fig=figure(c.plot,figsize=self.figsize)
                    if fig not in self.figures:
                        self.figures.append(fig)
                    
                    if c.plot>self.max_figure:
                        self.max_figure=c.plot
                    
                    if c.plot>=count:
                        clf()
                        count+=1
                        legends[count]=[]
                    legends[c.plot].append(c.label)
                    

                if isinstance(c.values,float):
                    c.values=c.values*ones(t.shape)
                h=plot(t,c.values,self.plot_style,label=c.label)
                color=h[0].get_color()
                
                if c.data:
                    if c.data['plot']:
                        plot(c.data['t'],c.data['value'],'s',color=color)
                        legends[c.plot].append(c.label+" data")
                
                
                pylab.xlabel(xlabel)
                pylab.grid(True)
                
                ylabel(c.label)
                gca().yaxis.set_major_formatter(ScalarFormatter(useOffset=False))

                drawit=True

        if count>1 and not self.noplots:
            
            for l in legends:
                if len(legends[l])>1:
                    figure(l,figsize=self.figsize)
                    legend(legends[l],loc='best')
                    ylabel('Value')
              
            if self.show:
                show()
            
        if drawit and not self.noplots:
            draw()

    def slider(self,**kwargs):
        from matplotlib.widgets import Slider
    
        if not self.max_figure:
            raise ValueError("No figures to adjust")
            
        def update(val):
            vals={}
            for key in self.sliders:
                vals[key]=self.sliders[key].val
            self.params(**vals)
            t_min,t_max=self['t'].min(),self['t'].max()
            self.run(t_min,t_max)
        
        params=kwargs
        L=len(params)
        figure(self.max_figure+1,figsize=(7,L))
        self.sliders={}

        for i,key in enumerate(sorted(params.keys())):
            ax=subplot(L,1,i+1)
            self.sliders[key] = Slider(ax, key, params[key][0], params[key][1], valinit=self[key])

            self.sliders[key].on_changed(update)
            show()
        
    def add_data(self,t,plot=False,**kwargs):
        for key in kwargs:
            c=self.get_component(key)
            c.data={'t':t,'plot':plot,'value':kwargs[key]}
            mx=max(t)
            if mx>self.maximum_data_t:
                self.maximum_data_t=mx


    def add_system(self,N,equations,initial_values,plot=True,**kwargs):
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
                
                self.add(eqn,val,plot=plot)

    def system_params(self,N,**kwargs):
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
            
        self.params(**params)
    

    def get_component(self,y):
    
        if isinstance(y,int):
            return self.components[y]
        else:
            found=[c for c in self.components+self.assignments if c.name==y]    
            return found[0]

    def __getattr__(self, item):
        """Maps values to attributes.
        Only called if there *isn't* an attribute with this name
        """
        try:
            return self.__getitem__(item)
        except KeyError:
            raise AttributeError(item)

    def __getitem__(self,y):
        import unicodedata
        
        try:
            return self.components[y].values
        except (TypeError,IndexError):
            if y=='t':
                return self.t
                
            found=[c for c in self.components+self.assignments if 
                            c.name==y or unicodedata.normalize('NFKC', c.name)==y]
            if found:
                return found[0].values

            if y in self.myparams:
                return self.myparams[y]
            else:
                raise IndexError("Unknown Index %s" % str(y))

def myfunc(t,p):
    if t<5:
        return 2*p*(1-p/5000.0)
    else:
        return -.3*p
        
def test1():
    sim=Simulation()
    sim.add("p'=a*p*(1-p/K)",100,plot=True)
    sim.params(a=1.5,K=300)
    
    sim.run(0,50)

def test2():
    sim=Simulation()
    sim.add("x=a*x*(1-x)",0.11,plot=1)
    sim.add("y=a*y*(1-y)",0.12,plot=1)
    sim.params(a=3.5)
    
    sim.run(0,50,discrete=True)
    
def test3():
    sim=Simulation('map')
    sim.add("x=a*x*(1-x)",0.11)

    for a in linspace(.1,4,200):
        sim.params(a=a)
        sim.run(0,1000)
    
        x=sim['x'][-100:]
        
        plot(a*ones(x.shape),x,'k.')
        
    #show()

def test4():
    sim=Simulation()
    sim.add("growth_rate=a*(1-p/K)",plot=True)
    sim.add("p'=growth_rate*p",100,plot=True)
    sim.params(a=1.5,K=300)
    
    sim.run(0,50)

def test_repeat():  # doesn't work yet
    sim=Simulation()
    sim.add("growth_rate=a*(1-p/K)")
    sim.add("p'=growth_rate*p",100)
    sim.params(a=1.5,K=300)

    sim.repeat(0,50,a=[1,2,3,4])

def test_higher_order():
    sim=Simulation()
    sim.add("x''=-k*x/m -b*x'",[10,0],plot=True)
    sim.params(k=1.0,m=1.0,b=0.5)
 
    sim.run(0,20)


def repeat(S_orig,t_min,t_max,**kwargs):
    keys=list(kwargs.keys())
    if kwargs:
        num_times=len(kwargs[keys[0]])
    else:
        num_times=1
        
    all_sims=[]
    for i in range(num_times):
        S=S_orig.copy()
        params=S.myparams  # parameter dictionary
    
        # update the parameters
        updated_params={}
        for k in list(kwargs.keys()):
            updated_params[k]=kwargs[k][i]
        params.update(updated_params)
 
        S.run(t_min,t_max)
        all_sims.append(S)
        
    return all_sims
        

def model(params,xd,sim,varname,parameters):
    cnames=[C.name for C in sim.components]
    pnames=[x[0] for x in parameters]
    for i,name in enumerate(pnames):
        if name in cnames:
            x=[C for C in sim.components if C.name==name][0]
            x.initial_value=params[i]
        else:
            d={name:params[i]}
            sim.params(**d)

    sim.run(0,xd.max())
    
    st=sim['t']
    svals=sim[varname]
    
    simvals=interp(xd,st,svals)
    
    return simvals
    
    
    
def mse_from_sim(params,extra):

    sim=extra['model']
    varname=extra['varname']
    parameters=extra['parameters']
    xd=extra['x']
    yd=extra['y']
    
    y=model(params,xd,sim,varname,parameters)
    
    mse=((array(yd)-y)**2).mean()
            
    return mse     



class particle(object):

    def __init__(self,parameters,fitness_function,extra=None):
        self.chi=0.72
        self.w=1.0
        self.c1=2.05
        self.c2=2.05
        self.extra=extra
        
        self.fitness_function=fitness_function
        self.dim=len(parameters)
        self.parameters=parameters
        self.parameter_names=[]
        self.x=[]
        self.v=[]
        for p in parameters:
            self.parameter_names.append(p[0])
            self.x.append(random.rand()*(p[1][1]-p[1][0])+p[1][0])
            self.v.append((2*random.rand()-1)*(p[1][1]-p[1][0]))

        self.x=array(self.x)
        self.v=array(self.v)

        self.best=self.x[:]
        
        if self.extra is None:
            self.best_val=self.fitness_function(self.x)
        else:
            self.best_val=self.fitness_function(self.x,self.extra)
        
    def fitness(self):
        if self.extra is None:
            val=self.fitness_function(self.x)
        else:
            val=self.fitness_function(self.x,self.extra)

        if val<self.best_val:
            self.best_val=val
            self.best=self.x[:]
            
        return val
        
    def update(self,global_best):
    
        r1=random.rand(len(self.x))
        r2=random.rand(len(self.x))
        
        
        self.v=self.chi*(self.w*self.v + 
                    self.c1*r1*(self.best-self.x)+
                    self.c2*r2*(global_best-self.x))
        self.x=self.x+self.v     
        
    def __repr__(self):
        s=""
        for i in range(self.dim):
            s+="(%s: %f) " % (self.parameter_names[i],self.x[i])
        return s
        
class swarm(object):

    def __init__(self,parameters,fitness,
                        number_of_particles=30,extra=None):
        
        self.particles=[particle(parameters,fitness,extra) 
                for i in range(number_of_particles)]

        self.best_val=1e500
        self.best=None
        
    def update(self):
    
        vals=[p.fitness() for p in self.particles]
            
        i=argmin(vals)    
        self.current_best=self.particles[i].x[:]
        self.current_val=vals[i]
        
        if self.current_val<self.best_val:
            self.best_val=self.current_val
            self.best=self.current_best[:]
            
        for p in self.particles:
            p.update(self.best)  
            
                
    def __repr__(self):
        return str(self.particles)
             
                
def pso_fit_sim(varname,xd,yd,sim,parameters,
        n_particles=30,n_iterations=-1,
        progress_interval=100,plot=False):

    extra={'x':xd,'y':yd,'model':sim,'varname':varname,
        'parameters':parameters}
        
    old_plots=[]    
    for c in sim.components:
        old_plots.append(c.plot)
        c.plot=False
        
    s=swarm(parameters,mse_from_sim,n_particles,extra)


    iterations=0
    stop=False
    try:
        while not stop:
            s.update()
            if iterations%progress_interval==0:
                print("iterations", iterations," min fitness: ",s.best_val, " with vals ",s.best)
                
                if plot:
                    pylab.clf()
                    
                    pylab.plot(xd,yd,'o-')        
                    
                    params=s.best
                    y=model(params,xd,sim,varname,parameters)

                    pylab.plot(xd,y,'-',linewidth=2)
                    
                    pylab.draw()
                    #pylab.show()
            iterations+=1
            if n_iterations>0 and iterations>=n_iterations:
                stop=True
    except KeyboardInterrupt:
        pass

    for c,op in zip(sim.components,old_plots):
        c.plot=op

    
    params=s.best
    params_dict={}
    for i,p in enumerate(parameters):
        params_dict[p[0]]=params[i]
    
    return params_dict

        
if __name__=="__main__":
    
    sim=Simulation()
    sim.add("growth_rate=a*(1-p/K)")
    sim.add("p'=growth_rate*p",100)
    sim.params(a=1.5,K=300)

#    sim.run(0,50)

    sims=repeat(sim,0,50,K=linspace(100,400,4))
    