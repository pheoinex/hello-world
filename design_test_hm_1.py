
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm


# In[56]:


#散点图
rain_data = pd.read_csv("data_rain.csv")
err_data = list(rain_data["y"] - 0.44*rain_data["x"])
x_lables = range(57)
x_data = list(rain_data["x"])
y_data = list(rain_data["y"])
#print(len(x_data))
plt.scatter(x_lables,err_data,s=10)
plt.title("Scatter picture",fontsize =24)
plt.xlabel("x_index",fontsize =14)
plt.ylabel("err",fontsize =14)


# In[63]:


#查找异常值
[err_data.index(i) for i in err_data if i>0.3]


# In[62]:


#去除异常值
x_data.pop(19)
x_data.pop(34)
#print(len(x_data))
y_data.pop(19)
y_data.pop(34)
#print(len(y_data))
plt.scatter(x_data,y_data,s=10)


# In[53]:


#加截距项的回归拟合
X_data = sm.add_constant(x_data)
est = sm.OLS(y_data,X_data)
est = est.fit()
est.summary()


# In[52]:


#不加截距项的回归拟合
est_1 = sm.OLS(y_data,x_data)
est_1 = est_1.fit()
est_1.summary()


# In[64]:


import numpy as np
import pandas as pd
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import statsmodels.api as sm
from scipy import stats
import matplotlib.pyplot as plt


# In[79]:


data_gun = pd.read_csv("gun_data.csv",skiprows = [0],names = ['area','speed','p1','p2','p3'])
data_gun
models = ols('speed ~ area',data_gun).fit()
anovat = anova_lm(models)
anovat
comp = pairwise_tukeyhsd(data_gun['speed'],data_gun['area'])
print(comp.summary())
help(pairwise_tukeyhsd)
data_speed = pd.read_csv("speed.csv")
X_3=np.mat(data_speed[['X0','X1','X2','X3']])
#X_5 =mat([0.016,0.03,0.044,0.058],[0.016,0.03,0.044,0.058],[0.016,0.03,0.044,0.058],[0.016,0.03,0.044,0.058])
X_d=X_3*(np.mat([[0.016,0.03,0.044,0.058],[0.016,0.03,0.044,0.058],[0.016,0.03,0.044,0.058],[0.016,0.03,0.044,0.058]]))
#print(X_d)

Y_1 = data_speed['Y']
X = sm.add_constant(X_d)
est_1 = sm.OLS(Y_1,X_d).fit()
est_1.summary()

