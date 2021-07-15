
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns
from numpy.linalg import inv
get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


ratings = pd.read_csv('/Users/wan/Desktop/bayesian/hw2/movies_csv/ratings.csv',names=('user_id','movie_id','rating'))
ratings_test = pd.read_csv('/Users/wan/Desktop/bayesian/hw2/movies_csv/ratings_test.csv',names=('user_id','movie_id','rating'))
import sys
f = open('/Users/wan/Desktop/bayesian/hw2/movies_csv/movies.txt')
movies = []
for line in f:
    movies.append(line)


# In[5]:


d= 5
c= 1
sigma= 1
sigma_2 = 1


# ## a)

# In[45]:


N = len(set(ratings['user_id']))
M = len(movies)
N,M


# In[46]:


R = np.zeros((N,M))
for row in ratings.itertuples():
    R[row[1]-1,row[2]-1]=row[3]


# In[77]:


def EM_update(U,V,R,c=1,sigma=1,sigma_2=1):
    """
    @ parameters
    U,V d*NUM 
    R rating matrix
    @ functions 
    
    """
    # U,V d*NUM 
    # R rating matrix
    d = U.shape[0]
    N = U.shape[1]
    M = V.shape[1]
    
    def estimate(U,V):
        # E_PHI matrix
        E_PHI = np.zeros((N,M))     
        UV_dot = np.dot(U.T,V)       
        norm_pdf = stats.norm.pdf(-UV_dot/sigma)
        norm_cdf = stats.norm.cdf(-UV_dot/sigma)
        for i in range(N):
            for j in range(M):
                if R[i][j]==1: 
                    E_PHI[i,j]= UV_dot[i,j]+sigma*norm_pdf[i,j]/(1-norm_cdf[i,j])
                elif R[i][j]==-1:
                    E_PHI[i,j]= UV_dot[i,j]-sigma*norm_pdf[i,j]/norm_cdf[i,j]
                elif R[i][j]==0:
                    continue
        return E_PHI


    def U_update(V,E_PHI):
        U_new = np.ones((d,N))
        for i in range(N):
            valid_V_index = np.where(R[i,:]!=0)[0]
            if len(valid_V_index)>0:
                ## calculate SUM(V*V.T) d-by-d
                V_2_sum = np.array([np.outer(V[:,j],V[:,j]) for j in valid_V_index]).sum(axis=0)
                
                factor1 = inv(1.0/c*np.identity(d)+1.0/sigma_2*V_2_sum)  #
                valid_V =  np.array([V[:,j] for j in valid_V_index]).T   
                valid_E_PHI = np.array([E_PHI[i,j] for j in valid_V_index]).reshape(-1,1)  # 1 column
                factor2 = np.dot(valid_V, valid_E_PHI)
                U_new[:,i] = (1.0/sigma_2)*np.dot(factor1,factor2).reshape(-1)
        return U_new
    
    
    def V_update(U,E_PHI):
        V_new = np.ones((d,M))
        for j in range(M):
            valid_U_index = np.where(R[:,j]!=0)[0]
            if len(valid_U_index)>0:
                ## calculate SUM(V*V.T) d-by-d
                U_2_sum = np.array([np.outer(U[:,i],U[:,i]) for i in valid_U_index]).sum(axis=0)
                
                factor1 = inv(1.0/c*np.identity(d)+1.0/sigma_2*U_2_sum) #
                valid_U =  np.array([U[:,i] for i in valid_U_index]).T   
                valid_E_PHI = np.array([E_PHI[i,j] for i in valid_U_index]).reshape(-1,1)  # 1 column
                factor2 = np.dot(valid_U, valid_E_PHI)
                V_new[:,j] = (1.0/sigma_2)*np.dot(factor1,factor2).reshape(-1)
        return V_new
    
    
    def cost(U,V):
        UV_dot = np.dot(U.T,V)
        norm_cdf = stats.norm.cdf(UV_dot/sigma)
        
        llh=0.0
        for i in range(N):
            for j in range(M):
                if R[i,j]==1:
                    llh+=np.log(norm_cdf[i,j])
                elif R[i,j]==-1:
                    llh+=np.log(1-norm_cdf[i,j])
                    
        U_sum=0.0
        V_sum=0.0
        for i in range(N):
            U_sum += np.dot(U[:,i],U[:,i])
        for j in range(M):
            V_sum += np.dot(V[:,j],V[:,j])
            
        fix = -(U_sum+V_sum)/(2*c)-1*(N+M)*d/2*np.log(2*np.pi*c)
            
        cost = fix+llh
        return cost
    
    """
    @ main part
    """
    E_PHI = estimate(U,V)
    U = U_update(V, E_PHI)
    E_PHI = estimate(U,V)
    V = V_update(U, E_PHI)
    Cost = cost(U,V)
    
    return (U, V, Cost)
    


# In[78]:


def EM(T=100):
    ## initialize u_i, v_i
    U = np.random.multivariate_normal(mean=np.zeros(d),cov=0.1*np.identity(d), size=N).T
    V = np.random.multivariate_normal(mean=np.zeros(d),cov=0.1*np.identity(d), size=M).T
    
    cost_list = []
    for t in range(T):
        U, V, Cost = EM_update(U, V, R)
        cost_list.append(Cost)
    return (U,V, cost_list)
        


# ## a)

# In[81]:


U, V, cost_list = EM(T=100)


# In[83]:


plt.plot(range(1,100),cost_list[1:])
plt.xlabel('Iteration')
plt.ylabel('Ln P(R,U,V)')


# ## b)

# In[84]:


U1, V1, cost_list1 = EM(T=100)


# In[85]:


U2, V2, cost_list2 = EM(T=100)


# In[86]:


U3, V3, cost_list3 = EM(T=100)


# In[87]:


U4, V4, cost_list4 = EM(T=100)


# In[88]:


U5, V5, cost_list5 = EM(T=100)


# In[90]:


plt.xlabel('Iteration')
plt.ylabel('Ln P(R,U,V)')

plt.plot(range(19,100),cost_list1[19:])
plt.plot(range(19,100),cost_list2[19:])
plt.plot(range(19,100),cost_list3[19:])
plt.plot(range(19,100),cost_list4[19:])
plt.plot(range(19,100),cost_list5[19:])
plt.legend(['1','2','3','4','5'])


# ## c)

# $$ Since\ r_{ij}|U,V\ \sim\ Bernoulli(\Phi(U_i^{T}V_j)),\ we\ can\ use\ the\ following\ prediction\ criteria:\\
#     r_{ij}=1\ if\ \Phi(U_i^{T}V_j)>=0.5\ or\ equally\ U_i^{T}V_j>=0\\
#     r_{ij}=-1\ if\ \Phi(U_i^{T}V_j)<0.5\ or\ equally\ U_i^{T}V_j<0\\
# $$

# In[91]:


def predict(U,V):
    R_pred = np.dot(U.T, V)/sigma
    R_pred[R_pred>=0]=1
    R_pred[R_pred<0]=-1
    return R_pred


# In[92]:


R_pred = predict(U,V)
test_pred = R_pred[ratings_test['user_id']-1,ratings_test['movie_id']-1 ]


# In[109]:


pd.crosstab(np.array(ratings_test['rating']), np.array(test_pred),
           rownames = ['True Ratings'],
           colnames = ['Predicted Ratings'])

