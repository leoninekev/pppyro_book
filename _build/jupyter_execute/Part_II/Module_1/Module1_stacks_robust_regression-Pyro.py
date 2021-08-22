#!/usr/bin/env python
# coding: utf-8

# ## Stacks: linear regression model

# In[1]:


import os
from functools import partial
import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly
import pyro
from scipy import stats
import pyro.distributions as dist
from torch import nn
from pyro.nn import PyroModule
from pyro.infer import MCMC, NUTS

import random
import numpy as np
import time
import random
import pandas as pd
from collections import defaultdict
import plotly.express as px
import re
import itertools

pyro.set_rng_seed(1)

# pd.options.plotting.backend = "plotly"
get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('default')


# **Stack Model definition**

# **beta0 ~ normal(0, 316);**
# 
# **beta ~ normal(0, 316);**
# 
# **sigmasq ~ inv_gamma(.001, .001);**

# In[2]:


def StackModel(X1, X2, X3, Y):
    beta0 = pyro.sample("beta0", dist.Normal(0., 316.))
    beta1 = pyro.sample("beta1", dist.Normal(0., 316))
    beta2 = pyro.sample("beta2", dist.Normal(0., 316))
    beta3 = pyro.sample("beta3", dist.Normal(0., 316))
    sigma = pyro.sample("sigma", dist.InverseGamma(0.001, 0.001))
    sigma = torch.sqrt(sigma)
    mu = beta0 + beta1 * X1 + beta2 * X2 + beta3*X3
    with pyro.plate("data", len(X1)):
        pyro.sample("obs", dist.Normal(mu, sigma), obs=Y)


# **Following standardises Data**

# In[3]:


transform_data= lambda x:torch.tensor(stats.zscore(x), dtype=torch.float)# standardises Input data

x_data= torch.tensor(np.array([80, 80, 75, 62, 62, 62, 62, 62, 59, 58, 58, 58, 58, 
58, 50, 50, 50, 50, 50, 56, 70, 27, 27, 25, 24, 22, 23, 24, 24, 
23, 18, 18, 17, 18, 19, 18, 18, 19, 19, 20, 20, 20, 89, 88, 90, 
87, 87, 87, 93, 93, 87, 80, 89, 88, 82, 93, 89, 86, 72, 79, 80, 
82, 91]).reshape((21,3)), dtype=torch.float)

y_data= torch.tensor([43, 37, 37, 28, 18, 18, 19, 20, 15, 14, 14, 13, 11, 12, 8, 7, 8, 8, 9, 15, 15], dtype=torch.float)

# X = X.reshape((3,21)).T
X1 = transform_data(x_data[:,0])
X2 = transform_data(x_data[:,1])
X3 = transform_data(x_data[:,2])
Y= y_data
print(X1.shape, X2.shape, X3.shape, Y.shape)


# In[4]:


def get_hmc_n_chains(num_chains=4, base_count = 900):
    hmc_sample_chains =defaultdict(dict)
    possible_samples_list= random.sample(list(np.arange(base_count, base_count+num_chains*100, 50)), num_chains)
    possible_burnin_list= random.sample(list(np.arange(100, 500, 50)), num_chains)

    t1= time.time()
    for idx, val in enumerate(list(zip(possible_samples_list, possible_burnin_list))):
        num_samples, burnin= val[0], val[1]
        nuts_kernel = NUTS(StackModel)
        mcmc = MCMC(nuts_kernel, num_samples=num_samples, warmup_steps=burnin)
        mcmc.run(X1,X2,X3,Y)
        hmc_sample_chains['chain_{}'.format(idx)]={k: v.detach().cpu().numpy() for k, v in mcmc.get_samples().items()}

    print("\nTotal time: ", time.time()-t1)
    hmc_sample_chains= dict(hmc_sample_chains)
    return hmc_sample_chains


# In[5]:


hmc_sample_chains= get_hmc_n_chains(num_chains=4, base_count = 900)


# **Parameter vs. Chain matrix**

# In[7]:


beta_chain_matrix_df = pd.DataFrame(hmc_sample_chains)
# beta_chain_matrix_df.to_csv("stack_regression_hmc_sample_chains.csv", index=False)
beta_chain_matrix_df


# **Key statistic results as dataframe**

# In[6]:


all_metric_func_map = lambda metric, vals: {"mean":np.mean(vals), "std":np.std(vals), 
                                            "25%":np.quantile(vals, 0.25), 
                                            "50%":np.quantile(vals, 0.50), 
                                            "75%":np.quantile(vals, 0.75)}.get(metric)


# In[10]:


key_metrics= ["mean", "std", "25%", "50%", "75%"]

summary_stats_df_= pd.DataFrame()
for metric in key_metrics:
    final_di = {}
    for column in beta_chain_matrix_df.columns:
        params_per_column_di = dict(beta_chain_matrix_df[column].apply(lambda x: all_metric_func_map(metric, x)))
        final_di[column]= params_per_column_di
    metric_df_= pd.DataFrame(final_di)
    metric_df_["parameter"]= metric
    summary_stats_df_= pd.concat([summary_stats_df_, metric_df_], axis=0)

summary_stats_df_.reset_index(inplace=True)
summary_stats_df_.rename(columns= {"index":"metric"}, inplace=True)
summary_stats_df_.set_index(["parameter", "metric"], inplace=True)
summary_stats_df_


# **Obtain 5 point Summary statics (mean, Q1-Q4, Std, ) as tabular data per chain.**
# 

# In[11]:


fit_df = pd.DataFrame()
for chain, values in hmc_sample_chains.items():
    param_df = pd.DataFrame(values)
    param_df["chain"]= chain
    fit_df= pd.concat([fit_df, param_df], axis=0)

fit_df.to_csv("data/stack_regression_hmc_samples.csv", index=False)    
fit_df


# In[8]:


# Use following once the results from pyro sampling operation are saved offline

# fit_df= pd.read_csv("data/stack_regression_hmc_samples.csv")
# fit_df


# In[9]:


summary_stats_df_2= pd.DataFrame()

for param in ["beta0", "beta1", "beta2", "beta3", "sigma"]:
    for name, groupdf in fit_df.groupby("chain"):
        groupdi = dict(groupdf[param].describe())

        values = dict(map(lambda key:(key, [groupdi.get(key)]), ['mean', 'std', '25%', '50%', '75%']))
        values.update({"parameter": param, "chain":name})
        summary_stats_df= pd.DataFrame(values)
        summary_stats_df_2= pd.concat([summary_stats_df_2, summary_stats_df], axis=0)
summary_stats_df_2.set_index(["parameter", "chain"], inplace=True)
summary_stats_df_2


# **Following Plots m parameters side by side for n chains**

# In[10]:


parameters= ["beta0", "beta1", "beta2", "beta3", "sigma"]# All parameters for given model
chains= fit_df["chain"].unique()# Number of chains sampled for given model


func_all_params_per_chain = lambda param, chain: (param, fit_df[fit_df["chain"]==chain][param].tolist())
func_all_chains_per_param = lambda chain, param: (f'{chain}', fit_df[param][fit_df["chain"]==chain].tolist())

di_all_params_per_chain = dict(map(lambda param: func_all_params_per_chain(param, "chain_0"), parameters))
di_all_chains_per_param = dict(map(lambda chain: func_all_chains_per_param(chain, "beta0"), chains))


# In[11]:


def plot_parameters_for_n_chains(chains=["chain_0"], parameters=["beta0", "beta1", "beta2", "beta3", "sigma"], plotting_cap=[4, 5], plot_interactive=False):
    try:
        chain_cap, param_cap = plotting_cap#
        assert len(chains)<=chain_cap, "Cannot plot Number of chains greater than %s!"%chain_cap
        assert len(parameters)<=param_cap, "Cannot plot Number of parameters greater than %s!"%param_cap
        
        for chain in chains:
            di_all_params_per_chain = dict(map(lambda param: func_all_params_per_chain(param, chain), parameters))
            df_all_params_per_chain = pd.DataFrame(di_all_params_per_chain)
            if df_all_params_per_chain.empty:
#                 raise Exception("Invalid chain number in context of model!")
                print("Note: Chain number [%s] is Invalid in context of this model!"%chain)
                continue
            if plot_interactive:
                df_all_params_per_chain= df_all_params_per_chain.unstack().reset_index(level=0)
                df_all_params_per_chain.rename(columns={"level_0":"parameters", 0:"values"}, inplace=True)
                fig = px.box(df_all_params_per_chain, x="parameters", y="values")
                fig.update_layout(height=600, width=900, title_text=f'{chain}')
                fig.show()
            else:
                df_all_params_per_chain.plot.box()
                plt.title(f'{chain}')
    except Exception as error:
        if type(error) is AssertionError:
            print("Note: %s"%error)
            chains = np.random.choice(chains, chain_cap, replace=False)
            parameters=np.random.choice(parameters, param_cap, replace=False)
            plot_parameters_for_n_chains(chains, parameters)
        else: print("Error: %s"%error)


# In[12]:


# Use plot_interactive=True for plotly plots offline

plot_parameters_for_n_chains(chains=['chain_0', 'chain_1', 'chain_2', 'chain_3'], parameters=parameters, plot_interactive=False)


# **Joint distribution of pair of each parameter sampled values**

# In[105]:


all_combination_params = list(itertools.combinations(parameters, 2))

for param_combo in all_combination_params:
    param1, param2= param_combo
    print("\nPyro -- %s"%(f'{param1} Vs. {param2}'))
    sns.jointplot(data=fit_df, x=param1, y=param2, hue= "chain")
    plt.title(f'{param1} Vs. {param2}')
    plt.show()
    


# **Pairplot distribution of each parameter with every other parameter's sampled values**

# In[106]:


sns.pairplot(data=fit_df, hue= "chain");


# **Hexbin plots**

# In[13]:


import matplotlib.pyplot as plt
import re
import itertools

get_ipython().run_line_magic('matplotlib', 'inline')
def hexbin_plot(x, y, x_label, y_label):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    min_x = min(list(x)+list(y)) - 0.1
    max_x = max(list(x)+list(y)) + 0.1
    ax.plot([min_x, max_x], [min_x, max_x])
    
    ax.set_xlim([min_x, max_x])
    ax.set_ylim([min_x, max_x])
    
    ax.set_title('{} vs. {} correlation scatterplot'.format(x_label, y_label))
    hbin= ax.hexbin(x, y, gridsize=25, mincnt=1, cmap=plt.cm.Reds)
    cb = fig.colorbar(hbin, ax=ax)
    cb.set_label('occurence_density')
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.show()
    
# hexbin_plot(difficulty_df_a['difficulty_keras'], difficulty_df_a['calibrated_difficulty'], "IRT difficulty for ch-a", "calibrated difficulty for ch-a")  


# In[14]:


fit_df


# In[15]:



def plot_interaction_hexbins(fit_df, parameters=["beta0", "beta1", "beta2", "beta3", "sigma"]):
    all_combination_params = list(itertools.combinations(parameters, 2))
    for param1, param2 in all_combination_params:#Plots interaction between each of two parameters
        hexbin_plot(fit_df[param1], fit_df[param2], param1, param2)
        
plot_interaction_hexbins(fit_df, parameters=["beta0", "beta1", "beta2", "beta3", "sigma"])


# **Loading Pystan model**

# **Pystan model for model comparison [For now Ignore]**

# In[16]:


import pystan as ps
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd
from plotly.subplots import make_subplots
# pd.options.plotting.backend = "plotly"


stacks_code = """
data {
  int<lower=0> N;
  int<lower=0> p;
  real Y[N];
  matrix[N,p] x;
} 

// to standardize the x's 
transformed data {
  matrix[N,p] z;
  row_vector[p] mean_x;
  real sd_x[p];
  for (j in 1:p) { 
    mean_x[j] <- mean(col(x,j)); 
    sd_x[j] <- sd(col(x,j)); 
    for (i in 1:N)  
      z[i,j] <- (x[i,j] - mean_x[j]) / sd_x[j]; 
  }
}

parameters {
  real beta0; 
  vector[p] beta; 
  real<lower=0> sigmasq; 
} 

transformed parameters {
  real<lower=0> sigma;
  vector[N] mu;
  sigma <- sqrt(sigmasq);
  mu <- beta0 + z * beta;
}

model {
  beta0 ~ normal(0, 316); 
  beta ~ normal(0, 316); 
  sigmasq ~ inv_gamma(.001, .001);
  Y ~ normal(mu, sigma);
} 

generated quantities {
  real b0;
  vector[p] b;
  real outlier_3;
  real outlier_4;
  real outlier_21;

  for (j in 1:p)
    b[j] <- beta[j] / sd_x[j];
  b0 <- beta0 - mean_x * b;

  outlier_3  <- step(fabs((Y[3] - mu[3]) / sigma) - 2.5);
  outlier_4  <- step(fabs((Y[4] - mu[4]) / sigma) - 2.5);
  outlier_21 <- step(fabs((Y[21] - mu[21]) / sigma) - 2.5);
}
"""

stacks_data = {"p":3, "N":21, "Y":[43, 37, 37, 28, 18, 18, 19, 20, 15, 14, 14, 13, 11, 12, 8, 
7, 8, 8, 9, 15, 15], "x":np.array([80, 80, 75, 62, 62, 62, 62, 62, 59, 58, 58, 58, 58, 
58, 50, 50, 50, 50, 50, 56, 70, 27, 27, 25, 24, 22, 23, 24, 24, 
23, 18, 18, 17, 18, 19, 18, 18, 19, 19, 20, 20, 20, 89, 88, 90, 
87, 87, 87, 93, 93, 87, 80, 89, 88, 82, 93, 89, 86, 72, 79, 80, 
82, 91]).reshape((21,3))}


# In[18]:


posterior = ps.StanModel(model_code=stacks_code)#, data=schools_data, random_seed=1)
fit = posterior.sampling(data=stacks_data, iter=1000, chains=4, seed=1)


# In[29]:


print(fit)


# In[61]:


params_pystan= fit.extract()
params_pystan


# In[22]:


fit_df_pystan= fit.to_dataframe()
fit_df_pystan.to_csv("data/stack_regression_hmc_samples[pystan].csv", index=False)
fit_df_pystan


# In[ ]:


# Use following once the results from Pystan sampling operation are saved offline

# fit_df_pystan= pd.read_csv("data/stack_regression_hmc_samples[pystan].csv")
# fit_df_pystan


# **Obtain 5 point Summary statics (mean, Q1-Q4, Std, ) as tabular data per chain.**
# 

# In[16]:


summary_stats_df_pystan= pd.DataFrame()

for param in ["beta0", "beta[1]", "beta[2]", "beta[3]"]:
    for name, groupdf in fit_df_pystan.groupby("chain"):
        groupdi = dict(groupdf[param].describe())
        
        values = dict(map(lambda key:(key, [groupdi.get(key)]), ['mean', 'std', '25%', '50%', '75%']))
        values.update({"parameter": param, "chain":name})
        summary_stats_df= pd.DataFrame(values)
        summary_stats_df_pystan= pd.concat([summary_stats_df_pystan, summary_stats_df], axis=0)
summary_stats_df_pystan.set_index(["parameter", "chain"], inplace=True)
summary_stats_df_pystan


# In[17]:


fit_df_pystan.head(5)


# **Following Plots m parameters side by side for n chains**

# In[23]:


parameters_pystan= ["beta0", "beta[1]", "beta[2]", "beta[3]"]# All parameters for given model
chains_pystan= fit_df_pystan["chain"].unique()# Number of chains sampled for given model


func_all_params_per_chain_pystan = lambda param, chain: (param, fit_df_pystan[fit_df_pystan["chain"]==chain][param].tolist())
func_all_chains_per_param_pystan = lambda chain, param: (f'Chain_{chain}', fit_df_pystan[param][fit_df_pystan["chain"]==chain].tolist())

di_all_params_per_chain_pystan = dict(map(lambda param: func_all_params_per_chain_pystan(param, 0), parameters_pystan))
di_all_chains_per_param_pystan = dict(map(lambda chain: func_all_chains_per_param_pystan(chain, "beta0"), chains_pystan))


# In[24]:



def plot_parameters_for_n_chains_pystan(chains=[0], parameters=["beta0", "beta[1]", "beta[2]", "beta[3]"], plotting_cap=[4, 4], plot_interactive=False):
    try:
        chain_cap, param_cap = plotting_cap#
        assert len(chains)<=chain_cap, "Cannot plot Number of chains greater than %s!"%chain_cap
        assert len(parameters)<=param_cap, "Cannot plot Number of parameters greater than %s!"%param_cap
        
        for chain in chains:
            di_all_params_per_chain = dict(map(lambda param: func_all_params_per_chain_pystan(param, chain), parameters))
            df_all_params_per_chain = pd.DataFrame(di_all_params_per_chain)
            if df_all_params_per_chain.empty:
#                 raise Exception("Invalid chain number in context of model!")
                print("Note: Chain number [%s] is Invalid in context of this model!"%chain)
                continue

            if plot_interactive:
              df_all_params_per_chain= df_all_params_per_chain.unstack().reset_index(level=0)
              df_all_params_per_chain.rename(columns={"level_0":"parameters", 0:"values"}, inplace=True)
              fig = px.box(df_all_params_per_chain, x="parameters", y="values")
              fig.update_layout(height=600, width=900, title_text=f'chain_{chain}')
              fig.show()
            else:
              df_all_params_per_chain.plot.box()
              plt.title(f'chain_{chain}')

    except Exception as error:
        if type(error) is AssertionError:
            print("Note: %s"%error)
            chains = np.random.choice(chains, chain_cap, replace=False)
            parameters=np.random.choice(parameters, param_cap, replace=False)
            plot_parameters_for_n_chains(chains, parameters)
        else: print("Error: %s"%error)


# In[25]:


# Use plot_interactive=True for plotly plots offline

plot_parameters_for_n_chains_pystan(chains=[0, 1, 3], parameters=["beta0", "beta[1]", "beta[2]", "beta[3]"], plot_interactive=False)


# **Following Plots n chains side by side for m parameters**

# In[26]:


def plot_chains_for_n_parameters_pystan(parameters=["beta0", "beta[1]", "beta[2]", "beta[3]"], chains=[0,1,2,3,4], plotting_cap=[4, 4], plot_interactive=False):
    screen_invalid_chains = lambda chain_val: True if not chain_val[1] else False# Filters Invalid chain results
    screen_valid_chain_results = lambda chain_val: True if chain_val[1] else False# Filters valid chain results
    try:
        chain_cap, param_cap = plotting_cap#
        assert len(chains)<=chain_cap, "Cannot plot Number of chains greater than %s!"%chain_cap
        assert len(parameters)<=param_cap, "Cannot plot Number of parameters greater than %s!"%param_cap
        
        for param in parameters:
#             di_all_params_per_chain = dict(map(lambda param: func_all_params_per_chain(param, chain), parameters))
            
            di_all_chains_per_param = dict(map(lambda chain: func_all_chains_per_param_pystan(chain, param), chains))

            invalid_chains = dict(filter(screen_invalid_chains, list(di_all_chains_per_param.items())))
            
            valid_chain_results = dict(filter(screen_valid_chain_results, list(di_all_chains_per_param.items())))
            
            df_all_chains_per_param = pd.DataFrame(valid_chain_results)

            
            if plot_interactive:
              df_all_chains_per_param= df_all_chains_per_param.unstack().reset_index(level=0)
              df_all_chains_per_param.rename(columns={"level_0":"chains", 0:"values"}, inplace=True)
              fig = px.box(df_all_chains_per_param, x="chains", y="values")
              fig.update_layout(height=600, width=900, title_text=f'parameter: {param}')
              fig.show()
            else:
              df_all_chains_per_param.plot.box()
              plt.title(f'parameter: {param}')
            
            # fig= df_all_chains_per_param.plot.box()# Uncomment if pandas backend is set to "plotly"
            # fig.update_layout(height=600, width=900, title_text=f'parameter: {param}')
            # fig.show()
            if invalid_chains:
                print("Note: Chain numbers %s are Invalid in context of this model!"%list(invalid_chains.keys()))
    except Exception as error:
        if type(error) is AssertionError:
            print("Note: %s"%error)
            chains = np.random.choice(chains, chain_cap, replace=False)
            parameters=np.random.choice(parameters, param_cap, replace=False)
            plot_chains_for_n_parameters(parameters, chains)
        else: print("Error: %s"%error)


# In[27]:


# Use plot_interactive=True for plotly plots offline


plot_chains_for_n_parameters_pystan(parameters=["beta0", "beta[1]", "beta[2]", "beta[3]"], chains=[0, 1, 3, 2], plot_interactive=False)


# In[28]:


# Uncomment following to view plotly plots offline

# di_all_chains_per_param_pystan = dict(map(lambda chain: func_all_chains_per_param_pystan(chain, param), chains_pystan))

# df_all_chains_per_param_pystan = pd.DataFrame(di_all_chains_per_param_pystan)
# df_all_chains_per_param_sqeezed_pystan = df_all_chains_per_param_pystan.melt(value_name='posterior').rename(columns={"variable":"chains"})

# fig_2 = px.box(df_all_chains_per_param_sqeezed_pystan, x="chains", y="posterior", points="all")
# fig_2.show()


# In[95]:


fit_df_pystan_trunc = fit_df_pystan[['chain', 'beta0', 'beta[1]', 'beta[2]', 'beta[3]', 'sigma']].copy()
fit_df_pystan_trunc["chain"]=fit_df_pystan_trunc["chain"].apply(lambda x: f'chain_{x}')
fit_df_pystan_trunc.head(3)


# **Joint distribution of pair of each parameter sampled values**

# In[107]:


all_combination_params = list(itertools.combinations(['beta0', 'beta[1]', 'beta[2]', 'beta[3]', 'sigma'], 2))

for param_combo in all_combination_params:
    param1, param2= param_combo
    print("\nPystan -- %s"%(f'{param1} Vs. {param2}'))
    sns.jointplot(data=fit_df_pystan_trunc, x=param1, y=param2, hue= "chain")
    plt.title(f'{param1} Vs. {param2}')
    plt.show()


# **Pairplot distribution of each parameter with every other parameter's sampled values**

# In[108]:


sns.pairplot(data=fit_df_pystan_trunc, hue= "chain");


# **Plot hexbin scatterplot for each parameter with respect to other parameters**

# In[60]:


params_pystan


# In[63]:



def plot_interaction_hexbins_pystan(params, parameters=['beta0', 'beta[1]', 'beta[2]', 'beta[3]']):
    all_param_keys_unison = "|".join((dict(params).keys()))# example - "beta0|beta|sigmasq|"
    filter_param_keys = lambda param: (param, re.findall(all_param_keys_unison, param)[0])

    parameters_to_select_param_map = dict(map(filter_param_keys, parameters))# outputs - {'beta0': 'beta0', 'beta[1]': 'beta', 'beta[2]': 'beta', 'beta[3]': 'beta'}

    select_param_keys= set(parameters_to_select_param_map.values())

    sliced_param_di = dict(map(lambda param: (param, dict(params).get(param)), select_param_keys))
    all_combination_params = list(itertools.combinations(parameters, 2))

    # parameters
    final_param_di= {}
    for param in sorted(parameters):
        arr= sliced_param_di[parameters_to_select_param_map.get(param)]
        if arr.ndim>1:# For params other than intercept/ bias/ beta0
            final_param_di[param]= arr[:, 0]
            arr= np.delete(arr, 0, 1)

        else: final_param_di[param]=arr
        sliced_param_di[parameters_to_select_param_map.get(param)]= arr


    for param1, param2 in all_combination_params:#Plots interaction between each of two parameters
        hexbin_plot(final_param_di[param1], final_param_di[param2], param1, param2)


# In[64]:


plot_interaction_hexbins_pystan(params_pystan, parameters=['beta0', 'beta[1]', 'beta[2]', 'beta[3]'])
#plot_interaction_hexbins(params, parameters=['beta[1]', 'beta[2]', 'beta[3]'])


# **Plotting Intermixing of Chains from Pyro & Pystan**

# In[31]:


# chains_pyro_pystan -- list of all chains from pyro & pystan 
chains_pyro_pystan = list(fit_df["chain"].unique())+ list(fit_df_pystan["chain"].unique())


# parameter_pyro_pystan_map -- map of each parameter between pyro & pystan
parameter_pyro_pystan_map = {'beta0': 'beta0', 'beta1': 'beta[1]', 'beta2': 'beta[2]','beta3': 'beta[3]', 'sigma': 'sigma'}

chains_pyro_pystan


# In[41]:


fit_df.head(2)


# In[39]:


fit_df_pystan.head(2)


# In[208]:


import matplotlib.pyplot as plt
import seaborn as sns

pyro_parameters_to_plot= ['beta0', 'beta1', 'beta2', 'beta3', 'sigma']# list(parameter_pyro_pystan_map.keys())
chains_to_plot = list(np.random.choice(chains_pyro_pystan, 5))
all_params_chains_df= pd.DataFrame()

sns.set_style("darkgrid")
fig= plt.figure(figsize=(7,7))
for param in pyro_parameters_to_plot:
    param_pystan= parameter_pyro_pystan_map.get(param)
    all_chains_df= pd.DataFrame()
    print("\nFor pyro_parameter: %s & pystan_parameter: %s"%(param, param_pystan))
    for chain in chains_to_plot:
        chain = int(chain) if chain.isnumeric() else chain
    #     if chain in list(fit_df_pystan["chain"].apply(str).unique()):
        if chain in list(fit_df_pystan["chain"].unique()):# Plots pystan chain
            all_chains_df_ = pd.DataFrame({param:fit_df_pystan[fit_df_pystan["chain"]==chain][param_pystan].tolist()})

            plt.plot(fit_df_pystan[fit_df_pystan["chain"]==chain][param_pystan])
#             plt.legend(f'pystan chain_{chain}', ncol=2, loc='upper left');
        else:
            all_chains_df_ =pd.DataFrame({param:fit_df[fit_df["chain"]==chain][param].tolist()})

            plt.plot(fit_df[fit_df["chain"]==chain][param])
#             plt.legend(f'pyro {chain}', ncol=2, loc='upper left');
        all_chains_df_["chain"] = chain
        all_chains_df =pd.concat([all_chains_df, all_chains_df_], axis=0)
#     all_chains_df["parameter"] = param
    all_params_chains_df= pd.concat([all_params_chains_df, all_chains_df], axis=1)

    plt.title("Sampled values for %s"%(param))
    plt.show()


# In[220]:


all_params_chains_df.head(2)


# In[221]:


all_params_chains_df_ = all_params_chains_df.copy()

all_params_chains_df_ = all_params_chains_df_.loc[:,~all_params_chains_df_.columns.duplicated()]
all_params_chains_df_.head(3)


# **Pairplot distribution of each parameter with every other parameter's sampled values**
# * Note: resulting density plots are from chains from both pyro & pystan sampling

# In[222]:


sns.pairplot(data=all_params_chains_df_, hue="chain");


# _______________
