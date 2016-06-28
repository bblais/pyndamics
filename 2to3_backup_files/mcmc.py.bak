import pymc
import os
import pylab as py
import numpy as np
import time
from copy import deepcopy
greek=['alpha','beta','gamma','chi','tau','sigma','lambda','epsilon','zeta','xi','theta','rho','psi']
import emcee


def histogram(y,bins=50,plot=True):
    N,bins=np.histogram(y,bins)
    
    dx=bins[1]-bins[0]
    if dx==0.0:  #  all in 1 bin!
        val=bins[0]
        bins=np.linspace(val-abs(val),val+abs(val),50)
        N,bins=np.histogram(y,bins)
    
    dx=bins[1]-bins[0]
    x=bins[0:-1]+(bins[1]-bins[0])/2.0
    
    y=N*1.0/np.sum(N)/dx
    
    if plot:
        py.plot(x,y,'o-')
        yl=py.gca().get_ylim()
        py.gca().set_ylim([0,yl[1]])
        xl=py.gca().get_xlim()
        if xl[0]<=0 and xl[0]>=0:    
            py.plot([0,0],[0,yl[1]],'k--')

    return x,y

def time2str(tm):
    
    frac=tm-int(tm)
    tm=int(tm)
    
    s=''
    sc=tm % 60
    tm=tm//60
    
    mn=tm % 60
    tm=tm//60
    
    hr=tm % 24
    tm=tm//24
    dy=tm

    if (dy>0):
        s=s+"%d d, " % dy

    if (hr>0):
        s=s+"%d h, " % hr

    if (mn>0):
        s=s+"%d m, " % mn


    s=s+"%.2f s" % (sc+frac)

    return s


def normal(x,mu,sigma):
    return 1/np.sqrt(2*np.pi*sigma**2)*np.exp(-(x-mu)**2/2.0/sigma**2)

def num2str(a):
    from numpy import abs
    if a==0:
        sa=''
    elif 0.001<abs(a)<10000:
        sa='%g' % a
    else:
        sa='%.3e' % a
        parts=sa.split('e')

        parts[1]=parts[1].replace('+00','')
        parts[1]=parts[1].replace('+','')
        parts[1]=parts[1].replace('-0','-')
        parts[1]=parts[1].replace('-0','')
        
        sa=parts[0]+r'\cdot 10^{%s}'%parts[1]
    
    return sa
    

class MCMCModel2(object):

    def __init__(self,sim,params,database=None,overwrite=False):
        self.sim=sim
        self.params=params
        self.model_dict=self.make_model_dict(self.sim,dict(self.params))
        self.MAP=pymc.MAP(self.model_dict)
        self.dbname=database
        if self.dbname is None:
            self.MCMC=pymc.MCMC(self.MAP.variables)
        else:
            base,ext=os.path.splitext(self.dbname)
            self.dbtype=ext[1:]
            
            if os.path.exists(self.dbname) and overwrite:
                print "Overwriting %s" % self.dbname
                os.remove(self.dbname)
                
            self.MCMC=pymc.MCMC(self.MAP.variables,db=self.dbtype,dbname=self.dbname)
        
            if self.MCMC.db.chains>0:
                self.set_mu()
        
        self.MCMCparams={}
        
        self.number_of_bins=200
        
        #self.fit(**kwargs)
        
    def sample(self,**kwargs):
        if not self.dbname is None:
            self.MCMC=pymc.MCMC(self.MAP.variables,db=self.dbtype,dbname=self.dbname)

        self.MCMC.sample(**kwargs)
        
        updated_params={}
        for key in self.params:
            self.__dict__[key]=self.MCMC.db.trace(key)[:]
            updated_params[key]=self.__dict__[key][-1]
            
        self.sim.params(**updated_params)

            
    def params_trace(self,N=None,*args):
        if len(args)==0:
            args=[key for key in self.params.keys() if not key.startswith('initial_')]
            
        key=args[0]
        L=len(self.model_dict[key].trace())
        if N is None:
            N=L
            
        for i in range(L-1,L-N,-1):
            d={}
            for key in args:
                d[key]=self.model_dict[key].trace()[i]
                
            yield d
        
    def draw(self):
        self.sample(iter=1,progress_bar=False)

    def plot(self,name,*args,**kwargs):
        if name in self.mu:
            try:
                all_chains=kwargs.pop('all_chains')
            except KeyError:
                all_chains=False

            try:
                show_95=kwargs.pop('show_95')
            except KeyError:
                show_95=False
                
            try:
                number_format=kwargs.pop('number_format')
            except KeyError:
                number_format='%.4g'

            try:
                show_normal=kwargs.pop('show_normal')
            except KeyError:
                show_normal=False

            try:
                show_mcmc=kwargs.pop('show_mcmc')
            except KeyError:
                show_mcmc=True
                
            if not show_normal and not show_mcmc:
                raise ValueError,"Move on...nothing to see here."

            x,y=self.get_distribution(name,all_chains,plot=True)
            py.plot(x,y,*args,**kwargs)

            mu=self.mu[name]
            sigma=self.sigma[name]
            percentile=self.percentile[name]
            
            
            if show_normal:
                xl=py.gca().get_xlim()
                xx=py.linspace(xl[0],xl[1],100)
                yy=normal(xx,mu,sigma)
                py.plot(xx,yy,*args,**kwargs)
            
            
            namestr=name
            for g in greek:
                if name.startswith(g):
                    namestr=r'\%s' % name
                        
            py.ylabel(r'$P(%s|{\rm data})$' % namestr)
            py.xlabel('$%s$' % namestr)
            
            yl=py.gca().get_ylim()
            xl=py.gca().get_xlim()
            if show_95:
                format_str=r'$\hat{%s}=%s (%s,%s) [95\%%%%]$' % (namestr,number_format,number_format,number_format)
                py.text(xl[0]+.05*(xl[1]-xl[0]),yl[1]*.8,format_str % (mu,percentile[0],percentile[1]))
            else:
                format_str=r'$\hat{%s}=%s \pm %s [1\sigma]$' % (namestr,number_format,number_format)            
                py.text(xl[0]+.05*(xl[1]-xl[0]),yl[1]*.8,format_str % (mu,sigma))
            py.grid(True)
            py.draw()

    def plot_distributions(self,number_format='%.4g',
                    separate_figures=True,figsize=(12,4),
                    show_normal=False,show_mcmc=True,show_95=False,all_chains=False):
        params=dict(self.params)
        keys=params.keys()

        if not separate_figures:
            py.figure(figsize=figsize)
        
        for i,p in enumerate(sorted(params)):
            if separate_figures:
                py.figure(figsize=figsize)
            else:
                py.subplot(len(params),1,i+1)
                
            self.plot(p,linewidth=3,number_format=number_format,
                    show_normal=show_normal,show_mcmc=show_mcmc,show_95=show_95,all_chains=all_chains)
            
        py.draw()
        
    def __getitem__(self,key):
        return self.model_dict[key]
        

    def set_mu(self):
        db=self.MCMC.db
    
        self.mu={}
        self.sigma={}
        self.percentile={}
        for key in self.params:
            values=db.trace(key)[:]
            
            self.mu[key]=np.median(values)   #mean(values)
            self.sigma[key]=(np.percentile(values,50+34.1)-np.percentile(values,50-34.1))/2   # std(values)
            self.percentile[key]=np.percentile(values,[2.5,97.5])
            self.__dict__[key]=float(self.model_dict[key].value)
      
            if key.startswith('initial_'):
                name=key.split('initial_')[1]
                _c=self.sim.get_component(name)
                _c.initial_value=self.mu[key]
                
            else:
                self.sim.params(**{key:self.mu[key]})
      
        variable=self.MCMC.get_node('std_dev')
        key='std_dev'
        self.mu[key]=np.mean(db.trace(key)[:])
        self.sigma[key]=np.std(db.trace(key)[:])
        self.percentile[key]=np.percentile(db.trace(key)[:],[2.5,97.5])
        self.__dict__[key]=float(variable.value)

    def fit(self,**kwargs):
        old_sim_noplots=self.sim.noplots
        self.sim.noplots=True
        t1=time.time()
    
        self.MAP.fit()
        
        
        self.MCMCparams=kwargs
        self.MCMCparams['iter']=self.MCMCparams.get('iter',50000)
        self.MCMCparams['burn']=self.MCMCparams.get('burn',self.MCMCparams['iter']//5)
        self.MCMC.sample(**self.MCMCparams)
        
        
        self.MAP.fit()
        
        t2=time.time()
        print
        print "Time taken:",time2str(t2-t1)

        self.set_mu()


        self.sim.noplots=old_sim_noplots
        
    def get_trace(self,name,all_chains=False):
        db=self.MCMC.db
        
        if all_chains:
            values=concatenate([db.trace(name,chain=c)[:] for c in range(db.chains)])
            print "%s: %d values" % (name,len(values))
        else:
            values=db.trace(name)[:]

        return values
        
    def get_distribution(self,name,all_chains=False,plot=True):
        db=self.MCMC.db
        
        if all_chains:
            values=np.concatenate([db.trace(name,chain=c)[:] for c in range(db.chains)])
            print "%s: %d values" % (name,len(values))
        else:
            values=db.trace(name)[:]
            
        self.mu[name]=np.median(values)   #mean(values)
        self.sigma[name]=(np.percentile(values,50+34.1)-np.percentile(values,50-34.1))/2   # std(values)
        self.percentile[name]=np.percentile(values,[2.5,97.5])

        mn=self.mu[name]-5*self.sigma[name]
        if mn<np.min(values):
            mn=np.min(values)

        mx=self.mu[name]+5*self.sigma[name]
        if mx>np.max(values):
            mx=np.max(values)

        
        bins=np.linspace(mn,mx,self.number_of_bins)
        
        x,y=histogram(values,bins,plot=plot)
        return x,y

        
    def reset(self,params):
        # this doesn't work
        self.sim=deepcopy(self.original_sim)

        self.model_dict=self.make_model_dict(self.sim,params)
        self.params=params
        self.MAP=pymc.MAP(self.model_dict)
        self.MCMC=pymc.MCMC(self.MAP.variables)
        
        return self.sim
           
    def plot_joint_distribution(self,var1,var2,show_prior=True,N=10000,fit_contour=False,all_chains=False):
        import scipy.stats
        model=self        
        db=self.MCMC.db
        
        if all_chains:
            values1=np.concatenate([db.trace(var1,chain=c)[:] for c in range(db.chains)])
            values2=np.concatenate([db.trace(var2,chain=c)[:] for c in range(db.chains)])
            print "%s: %d values" % (name,len(values1))
        else:
            values1=db.trace(var1)[:]
            values2=db.trace(var2)[:]
        
        
        if show_prior:            
            py.plot([model[var1].random() for i in range(N)],
                 [model[var2].random() for i in range(N)],
                 linestyle='none', marker='o', color='blue', mec='blue',
                 alpha=.5, label='Prior', zorder=-100)
                 
        py.plot(values1[:N], values2[:N],
             linestyle='none', marker='o', color='green', mec='green',
             alpha=.5, label='Posterior', zorder=-99)
        
        if fit_contour:
            xl=py.gca().get_xlim()
            yl=py.gca().get_ylim()
            gkde = scipy.stats.gaussian_kde([values1, values2])


            x=np.linspace(xl[0],xl[1],100)
            y=np.linspace(yl[0],yl[1],100)
            ix,iy=np.indices([len(x),len(y)])
            x=x[ix]
            y=y[iy]
            
            cmap=py.cm.Greys
            cmap=py.cm.jet
            z = np.array(gkde.evaluate([x.flatten(),y.flatten()])).reshape(x.shape)
            py.contour(x, y, z, linewidths=1, alpha=.5, cmap=cmap)
            
            


        name=var1
        namestr=name
        for g in greek:
            if name.startswith(g):
                namestr=r'\%s' % name
                    
        py.xlabel('$%s$' % namestr)

        name=var2
        namestr=name
        for g in greek:
            if name.startswith(g):
                namestr=r'\%s' % name
                    
        py.ylabel('$%s$' % namestr)

        if show_prior:
            py.legend()
                
    def make_model_dict(self,sim,params):
        initial_values={}
        initial_components={}
        model_dict={}
    
        params['std_dev']=params.get('std_dev',[1e-3,200])
        std_dev=pymc.Uniform('std_dev', params['std_dev'][0], params['std_dev'][1])  
        del params['std_dev']
    
    
        @pymc.deterministic(plot=False)
        def precision(std_dev=std_dev):
            return 1.0 / (std_dev * std_dev)
    
        for key in params:
            if key.startswith('initial_'):
                if params[key][0] is None:
                    initial_values[key]=pymc.Uninformative(key,value=params[key][1])
                else:
                    if len(params[key])==2:
                        initial_values[key]=pymc.Uniform(key,params[key][0],params[key][1])
                    elif len(params[key])==3:
                        initial_values[key]=pymc.Uniform(key,params[key][0],params[key][1],value=params[key][2])
                    else:
                        raise ValueError                
                
                name=key.split('initial_')[1]
                _c=sim.get_component(name)
                initial_components[key]=_c
            else:
                if params[key][0] is None:
                    params[key]=pymc.Uninformative(key,value=params[key][1])
                else:
                    if len(params[key])==2:
                        params[key]=pymc.Uniform(key,params[key][0],params[key][1])
                    elif len(params[key])==3:
                        params[key]=pymc.Uniform(key,params[key][0],params[key][1],value=params[key][2])
                    else:
                        raise ValueError                
           
        for key in initial_values:
            del params[key]
            
        def runit(**kwargs):
            sim.params(**kwargs)
            for key in initial_values:
                initial_components[key].initial_value=initial_values[key]
            
            try:
                sim.run_fast()
                return 0
            except FloatingPointError:
                return -1
    
        run_sim = pymc.Deterministic( eval = runit, 
                        doc="this",name = 'run_sim',
                        parents = dict(params.items()+initial_values.items()))
    
        def make_fun(var):
            def fun(run_sim=run_sim,sim=sim):
                _c=sim.get_component(var)
                if _c.data:
                    t=_c.data['t']
                    value=sim.interpolate(t,var)
                else:
                    t=sim.maximum_t
                    value=sim.interpolate(t,var)
                return value
        
        
            return fun
    
        for _c in sim.components+sim.assignments:
            params[_c.name]=pymc.Deterministic( eval=make_fun(_c.name),
                         doc=_c.name,
                         name=_c.name,
                        parents={'run_sim':run_sim,'sim':sim})
            if _c.data:
                varname=_c.name+'_data'
                params[varname]=pymc.Normal(varname,
                                mu=params[_c.name],
                                tau=precision,
                                observed=True,
                                value=_c.data['value'])
            
    
        model_dict.update(initial_values)
        model_dict.update(params)
    
        for key in ['std_dev','precision','run_sim']:
            model_dict[key]=locals()[key]

        return model_dict
    


class MCMCModel3(object):

    def __init__(self,sim,params,database=None,overwrite=False):
        raise ValueError,'database not implemented yet'
    
        self.sim=sim
        self.params=params
        self.model=self.make_model(self.sim,dict(self.params))
        with model:
            self.MAP = pymc.find_MAP()
            self.step = pymc.NUTS()
            
            
        self.dbname=database
        if self.dbname is None:
            pass
        else:
            raise ValueError,'database not implemented yet'
        
        self.MCMCparams={}
        
        self.number_of_bins=200
        
        
    def sample(self,**kwargs):
        if not self.dbname is None:
            raise ValueError,'database not implemented yet'

        self.trace = pm.sample(3000, self.step, self.MAP,**kwargs)
        
        updated_params={}
        for key in self.params:
            self.__dict__[key]=self.trace[key][:]
            updated_params[key]=self.__dict__[key][-1]
            
        self.sim.params(**updated_params)

            
    def params_trace(self,N=None,*args):
        raise ValueError,'not implemented yet'
        
        if len(args)==0:
            args=[key for key in self.params.keys() if not key.startswith('initial_')]
            
        key=args[0]
        L=len(self.model_dict[key].trace())
        if N is None:
            N=L
            
        for i in range(L-1,L-N,-1):
            d={}
            for key in args:
                d[key]=self.model_dict[key].trace()[i]
                
            yield d
        
    def draw(self):
        raise ValueError,'not implemented yet'
        self.sample(iter=1,progress_bar=False)

    def plot(self,name,*args,**kwargs):
        raise ValueError,'not implemented yet'
        if name in self.mu:
            try:
                all_chains=kwargs.pop('all_chains')
            except KeyError:
                all_chains=False

            try:
                show_95=kwargs.pop('show_95')
            except KeyError:
                show_95=False
                
            try:
                number_format=kwargs.pop('number_format')
            except KeyError:
                number_format='%.4g'

            try:
                show_normal=kwargs.pop('show_normal')
            except KeyError:
                show_normal=False

            try:
                show_mcmc=kwargs.pop('show_mcmc')
            except KeyError:
                show_mcmc=True
                
            if not show_normal and not show_mcmc:
                raise ValueError,"Move on...nothing to see here."

            x,y=self.get_distribution(name,all_chains,plot=True)
            py.plot(x,y,*args,**kwargs)

            mu=self.mu[name]
            sigma=self.sigma[name]
            percentile=self.percentile[name]
            
            
            if show_normal:
                xl=py.gca().get_xlim()
                xx=py.linspace(xl[0],xl[1],100)
                yy=normal(xx,mu,sigma)
                py.plot(xx,yy,*args,**kwargs)
            
            
            namestr=name
            for g in greek:
                if name.startswith(g):
                    namestr=r'\%s' % name
                        
            py.ylabel(r'$P(%s|{\rm data})$' % namestr)
            py.xlabel('$%s$' % namestr)
            
            yl=py.gca().get_ylim()
            xl=py.gca().get_xlim()
            if show_95:
                format_str=r'$\hat{%s}=%s (%s,%s) [95\%%%%]$' % (namestr,number_format,number_format,number_format)
                py.text(xl[0]+.05*(xl[1]-xl[0]),yl[1]*.8,format_str % (mu,percentile[0],percentile[1]))
            else:
                format_str=r'$\hat{%s}=%s \pm %s [1\sigma]$' % (namestr,number_format,number_format)            
                py.text(xl[0]+.05*(xl[1]-xl[0]),yl[1]*.8,format_str % (mu,sigma))
            py.grid(True)
            py.draw()

    def plot_distributions(self,number_format='%.4g',
                    separate_figures=True,figsize=(12,4),
                    show_normal=False,show_mcmc=True,show_95=False,all_chains=False):
        raise ValueError,'not implemented yet'
        params=dict(self.params)
        keys=params.keys()

        if not separate_figures:
            py.figure(figsize=figsize)
        
        for i,p in enumerate(sorted(params)):
            if separate_figures:
                py.figure(figsize=figsize)
            else:
                py.subplot(len(params),1,i+1)
                
            self.plot(p,linewidth=3,number_format=number_format,
                    show_normal=show_normal,show_mcmc=show_mcmc,show_95=show_95,all_chains=all_chains)
            
        py.draw()
        
    def __getitem__(self,key):
        return self.model_dict[key]
        

    def set_mu(self):
        raise ValueError,'not implemented yet'
        db=self.MCMC.db
    
        self.mu={}
        self.sigma={}
        self.percentile={}
        for key in self.params:
            values=db.trace(key)[:]
            
            self.mu[key]=np.median(values)   #mean(values)
            self.sigma[key]=(np.percentile(values,50+34.1)-np.percentile(values,50-34.1))/2   # std(values)
            self.percentile[key]=np.percentile(values,[2.5,97.5])
            self.__dict__[key]=float(self.model_dict[key].value)
      
            if key.startswith('initial_'):
                name=key.split('initial_')[1]
                _c=self.sim.get_component(name)
                _c.initial_value=self.mu[key]
                
            else:
                self.sim.params(**{key:self.mu[key]})
      
        variable=self.MCMC.get_node('std_dev')
        key='std_dev'
        self.mu[key]=np.mean(db.trace(key)[:])
        self.sigma[key]=np.std(db.trace(key)[:])
        self.percentile[key]=np.percentile(db.trace(key)[:],[2.5,97.5])
        self.__dict__[key]=float(variable.value)

    def fit(self,**kwargs):
        raise ValueError,'not implemented yet'
        old_sim_noplots=self.sim.noplots
        self.sim.noplots=True
        t1=time.time()
    
        self.MAP.fit()
        
        
        self.MCMCparams=kwargs
        self.MCMCparams['iter']=self.MCMCparams.get('iter',50000)
        self.MCMCparams['burn']=self.MCMCparams.get('burn',self.MCMCparams['iter']//5)
        self.MCMC.sample(**self.MCMCparams)
        
        
        self.MAP.fit()
        
        t2=time.time()
        print
        print "Time taken:",time2str(t2-t1)

        self.set_mu()


        self.sim.noplots=old_sim_noplots
        
    def get_trace(self,name,all_chains=False):
        raise ValueError,'not implemented yet'
        db=self.MCMC.db
        
        if all_chains:
            values=concatenate([db.trace(name,chain=c)[:] for c in range(db.chains)])
            print "%s: %d values" % (name,len(values))
        else:
            values=db.trace(name)[:]

        return values
        
    def get_distribution(self,name,all_chains=False,plot=True):
        raise ValueError,'not implemented yet'
        db=self.MCMC.db
        
        if all_chains:
            values=np.concatenate([db.trace(name,chain=c)[:] for c in range(db.chains)])
            print "%s: %d values" % (name,len(values))
        else:
            values=db.trace(name)[:]
            
        self.mu[name]=np.median(values)   #mean(values)
        self.sigma[name]=(np.percentile(values,50+34.1)-np.percentile(values,50-34.1))/2   # std(values)
        self.percentile[name]=np.percentile(values,[2.5,97.5])

        mn=self.mu[name]-5*self.sigma[name]
        if mn<np.min(values):
            mn=np.min(values)

        mx=self.mu[name]+5*self.sigma[name]
        if mx>np.max(values):
            mx=np.max(values)

        
        bins=np.linspace(mn,mx,self.number_of_bins)
        
        x,y=histogram(values,bins,plot=plot)
        return x,y

        
    def reset(self,params):
        raise ValueError,'not implemented yet'
        # this doesn't work
        self.sim=deepcopy(self.original_sim)

        self.model_dict=self.make_model_dict(self.sim,params)
        self.params=params
        self.MAP=pymc.MAP(self.model_dict)
        self.MCMC=pymc.MCMC(self.MAP.variables)
        
        return self.sim
           
    def plot_joint_distribution(self,var1,var2,show_prior=True,N=10000,fit_contour=False,all_chains=False):
        raise ValueError,'not implemented yet'
        import scipy.stats
        model=self        
        db=self.MCMC.db
        
        if all_chains:
            values1=np.concatenate([db.trace(var1,chain=c)[:] for c in range(db.chains)])
            values2=np.concatenate([db.trace(var2,chain=c)[:] for c in range(db.chains)])
            print "%s: %d values" % (name,len(values1))
        else:
            values1=db.trace(var1)[:]
            values2=db.trace(var2)[:]
        
        
        if show_prior:            
            py.plot([model[var1].random() for i in range(N)],
                 [model[var2].random() for i in range(N)],
                 linestyle='none', marker='o', color='blue', mec='blue',
                 alpha=.5, label='Prior', zorder=-100)
                 
        py.plot(values1[:N], values2[:N],
             linestyle='none', marker='o', color='green', mec='green',
             alpha=.5, label='Posterior', zorder=-99)
        
        if fit_contour:
            xl=py.gca().get_xlim()
            yl=py.gca().get_ylim()
            gkde = scipy.stats.gaussian_kde([values1, values2])


            x=np.linspace(xl[0],xl[1],100)
            y=np.linspace(yl[0],yl[1],100)
            ix,iy=np.indices([len(x),len(y)])
            x=x[ix]
            y=y[iy]
            
            cmap=py.cm.Greys
            cmap=py.cm.jet
            z = np.array(gkde.evaluate([x.flatten(),y.flatten()])).reshape(x.shape)
            py.contour(x, y, z, linewidths=1, alpha=.5, cmap=cmap)
            
            


        name=var1
        namestr=name
        for g in greek:
            if name.startswith(g):
                namestr=r'\%s' % name
                    
        py.xlabel('$%s$' % namestr)

        name=var2
        namestr=name
        for g in greek:
            if name.startswith(g):
                namestr=r'\%s' % name
                    
        py.ylabel('$%s$' % namestr)

        if show_prior:
            py.legend()
                
    def make_model(self,sim,params):
        initial_values={}
        initial_components={}
        model_dict={}

        def runit(**kwargs):
            sim.params(**kwargs)
            for key in initial_values:
                initial_components[key].initial_value=initial_values[key]
            
            try:
                sim.run_fast()
                return 0
            except FloatingPointError:
                return -1
    

        
        params['std_dev']=params.get('std_dev',[1e-3,200])
        with pymc.Model() as model:    
            std_dev=pymc.Uniform('std_dev', params['std_dev'][0], params['std_dev'][1])  
            del params['std_dev']
    
#         @pymc.deterministic(plot=False)
#         def precision(std_dev=std_dev):
#             return 1.0 / (std_dev * std_dev)
# 
            for key in params:
                if key.startswith('initial_'):
                    if params[key][0] is None:
                        initial_values[key]=pymc.Uninformative(key,value=params[key][1])
                    else:
                        if len(params[key])==2:
                            initial_values[key]=pymc.Uniform(key,params[key][0],params[key][1])
                        elif len(params[key])==3:
                            initial_values[key]=pymc.Uniform(key,params[key][0],params[key][1],value=params[key][2])
                        else:
                            raise ValueError                
                
                    name=key.split('initial_')[1]
                    _c=sim.get_component(name)
                    initial_components[key]=_c
                else:
                    if params[key][0] is None:
                        params[key]=pymc.Uninformative(key,value=params[key][1])
                    else:
                        if len(params[key])==2:
                            params[key]=pymc.Uniform(key,params[key][0],params[key][1])
                        elif len(params[key])==3:
                            params[key]=pymc.Uniform(key,params[key][0],params[key][1],value=params[key][2])
                        else:
                            raise ValueError                
           
            for key in initial_values:
                del params[key]
            
            run_sim = pymc.Deterministic( eval = runit, 
                            doc="this",name = 'run_sim',
                            parents = dict(params.items()+initial_values.items()))
    
            def make_fun(var):
                def fun(run_sim=run_sim,sim=sim):
                    _c=sim.get_component(var)
                    if _c.data:
                        t=_c.data['t']
                        value=sim.interpolate(t,var)
                    else:
                        t=sim.maximum_t
                        value=sim.interpolate(t,var)
                    return value
        
        
                return fun
    
            for _c in sim.components+sim.assignments:
                params[_c.name]=pymc.Deterministic( eval=make_fun(_c.name),
                             doc=_c.name,
                             name=_c.name,
                            parents={'run_sim':run_sim,'sim':sim})
                if _c.data:
                    varname=_c.name+'_data'
                    params[varname]=pymc.Normal(varname,
                                    mu=params[_c.name],
                                    tau=1.0/std_dev**2,
                                    observed=True,
                                    value=_c.data['value'])
            
    

        return model
    

if pymc.__version__[0]=='2':
    MCMCModel=MCMCModel2
elif pymc.__version__[0]=='3':
    MCMCModel=MCMCModel3
else:
    raise ValueError,"Bad Bad Bad"
    
    