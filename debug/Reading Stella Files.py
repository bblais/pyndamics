
# coding: utf-8

# http://www.iseesystems.com/solutions/systems-in-focus.aspx
# 
# <img src="http://www.iseesystems.com/images/solutions/energy-lg.jpg">

# In[2]:

fname='SiF-energy.stmx'
from webutils import Soup


# In[3]:

text=Soup(fname)


# In[26]:

stocks=text('stock')
for stock in stocks:
    name=stock.get('name')
    if name:
        name=name.replace('\\n','_')
        print name


# In[31]:

stock=stocks[0]
print stock


# In[32]:

stock.get('name')


# In[38]:

stock('eqn')[0].contents[0]


# In[ ]:



