#!/usr/bin/env python
# coding: utf-8

# ## Dogs: loglinear model for binary data
# 
#     Following is Bayesian modelling of Lindley's analysis of dogs data from Solomon-Wynne experiment in Pyro.
#     The experiment intended to study the learning hailing from past traumatic experiences in dogs and reach a plausible model where dogs learn to avoid the scenerios responsible for causing trauma in past (Here, avoiding jumping off the barriers loaded with electric shocks).
# 
# 
# 
# 
# 
# Following analysis uses a model
# 
# 
# $$
# \pi_j   =   A^{xj} B^{j-xj}
# $$
# 
# $$
# Wherein\ dog\ learns\ from\ previous\ trials.\\
# Here\ \pi_j\ is\ the\ probability\ of\ a\ shock\ at\ trial\ j\ depends\ on\ the\ number\ of\ previous\ shocks\ and\ the\ number\ of\ previous\ avoidances\ by\ the\ dog.
# $$

# In[1]:


import os
import torch
import pyro
import random
import time
import numpy as np
import pandas as pd
import re
import itertools
import seaborn as sns
import matplotlib.pyplot as plt
from functools import partial
from collections import defaultdict

from scipy import stats
import pyro.distributions as dist
from torch import nn
from pyro.nn import PyroModule
from pyro.infer import MCMC, NUTS


import plotly
import plotly.express as px
import plotly.figure_factory as ff

pyro.set_rng_seed(1)

# Uncomment following if pandas available with plotly backend

# pd.options.plotting.backend = "plotly"

get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('default')


# ### 1. Model Specification: Dogs Model definition
# ________
#   
#   **BUGS model**
#   
# $\log\pi_j = \alpha\ x_j + \beta\ ( $j$-x_j )$
# 
#    **Here**
#    * $\log\pi_j$ is log probability of a dog getting shocked at trial $j$
#    * $x_j$ is number of successful avoidances of shock prior to trial $j$.
#    * $j-x_j$ is number of shocks experienced prior to trial $j$.
#    *  $\alpha$ is the coefficient corresponding to number of success, $\beta$ is the coefficient corresponding to number of failures.
# 
#   
#   ____________________
#   
#   **Equivalent Stan model** 
#   
#       {
#   
#       alpha ~ normal(0.0, 316.2);
#   
#       beta  ~ normal(0.0, 316.2);
#   
#       for(dog in 1:Ndogs)
#   
#         for (trial in 2:Ntrials)  
# 
#           y[dog, trial] ~ bernoulli(exp(alpha * xa[dog, trial] + beta * 
#           xs[dog, trial]));
#       
#       }  
# 

# **Bugs model implementation**

# In[2]:


# Dogs model with normal prior
def DogsModel(x_avoidance, x_shocked, y):
    """
      model {
      alpha ~ normal(0.0, 316.2);
      beta  ~ normal(0.0, 316.2);
      for(dog in 1:Ndogs)  
        for (trial in 2:Ntrials)  
          y[dog, trial] ~ bernoulli(exp(alpha * xa[dog, trial] + beta * xs[dog, trial]));
    """
    alpha = pyro.sample("alpha", dist.Normal(0., 316.))
    beta = pyro.sample("beta", dist.Normal(0., 316))
    with pyro.plate("data"):
        pyro.sample("obs", dist.Bernoulli(torch.exp(alpha*x_avoidance + beta * x_shocked)), obs=y)


# Dogs model with uniform prior
def DogsModelUniformPrior(x_avoidance, x_shocked, y):
    """
      model {
      alpha ~ uniform(-10, -0.00001);
      beta  ~ uniform(-10, -0.00001);
      for(dog in 1:Ndogs)  
        for (trial in 2:Ntrials)  
          y[dog, trial] ~ bernoulli(exp(alpha * xa[dog, trial] + beta * xs[dog, trial]));
    """
    alpha = pyro.sample("alpha", dist.Uniform(-10, -0.00001))
    beta = pyro.sample("beta", dist.Uniform(-10, -0.00001))
    with pyro.plate("data"):
        pyro.sample("obs", dist.Bernoulli(torch.exp(alpha*x_avoidance + beta * x_shocked)), obs=y)
        


# **Following processes target label `y` to obtain input data `x_avoidance` & `x_shocked` where:**
# * `x_avoidance` :  number of shock avoidances before current trial.
# * `x_shocked` :  number of shocks before current trial.

# In[3]:


def transform_data(Ndogs=30, Ntrials=25, Y= np.array([0, 0, 0, 0])):
    y= np.zeros((Ndogs, Ntrials))
    xa= np.zeros((Ndogs, Ntrials))
    xs= np.zeros((Ndogs, Ntrials))

    for dog in range(Ndogs):
        for trial in range(1, Ntrials+1):
            xa[dog, trial-1]= np.sum(Y[dog, :trial-1]) #Number of successful avoidances uptill previous trial
            xs[dog, trial-1]= trial -1 - xa[dog, trial-1] #Number of shocks uptill previous trial
    for dog in range(Ndogs):
        for trial in range(Ntrials):
            y[dog, trial]= 1- Y[dog, trial]
    xa= torch.tensor(xa, dtype=torch.float)
    xs= torch.tensor(xs, dtype=torch.float)  
    y= torch.tensor(y, dtype=torch.float)

    return xa, xs, y


# In[4]:


dogs_data = {"Ndogs":30, 
             "Ntrials":25, 
             "Y": np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 
                  0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                  0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 
                  0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 
                  1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 
                  1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 
                  0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 
                  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 
                  0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
                  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 
                  1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
                  0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
                  1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 
                  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
                  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
                  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
                  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
                  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]).reshape((30,25))}


# In[5]:



x_avoidance, x_shocked, y= transform_data(**dogs_data)
print("x_avoidance: %s, x_shocked: %s, y: %s"%(x_avoidance.shape, x_shocked.shape, y.shape))

print("\nSample x_avoidance: %s \n\nSample x_shocked: %s"%(x_avoidance[1], x_shocked[1]))


# ### 2. Prior predictive checking

# In[6]:


priors_list= [(pyro.sample("alpha", dist.Normal(0., 316.)).item(), 
               pyro.sample("beta", dist.Normal(0., 316.)).item()) for index in range(1100)]# Picking 1100 prior samples

prior_samples = {"alpha":list(map(lambda prior_pair:prior_pair[0], priors_list)), "beta":list(map(lambda prior_pair:prior_pair[1], priors_list))}


# In[7]:


title= "parameter distribution for : %s"%(chain)
fig = ff.create_distplot(list(prior_samples.values()), list(prior_samples.keys()))
fig.update_layout(title="Prior distribution of parameters", xaxis_title="parameter values", yaxis_title="density", legend_title="parameters")
fig.show()

print("Prior alpha Q(0.5) :%s | Prior beta Q(0.5) :%s"%(np.quantile(prior_samples["alpha"], 0.5), np.quantile(prior_samples["beta"], 0.5)))


# ### 3. Posterior estimation

# In[6]:


def get_hmc_n_chains(pyromodel, xa, xs, y, num_chains=4, base_count = 900):
    hmc_sample_chains =defaultdict(dict)
    possible_samples_list= random.sample(list(np.arange(base_count, base_count+num_chains*100, 50)), num_chains)
    possible_burnin_list= random.sample(list(np.arange(100, 500, 50)), num_chains)

    t1= time.time()
    for idx, val in enumerate(list(zip(possible_samples_list, possible_burnin_list))):
        num_samples, burnin= val[0], val[1]
        nuts_kernel = NUTS(pyromodel)
        mcmc = MCMC(nuts_kernel, num_samples=num_samples, warmup_steps=burnin)
        mcmc.run(xa, xs, y)
        hmc_sample_chains['chain_{}'.format(idx)]={k: v.detach().cpu().numpy() for k, v in mcmc.get_samples().items()}

    print("\nTotal time: ", time.time()-t1)
    hmc_sample_chains= dict(hmc_sample_chains)
    return hmc_sample_chains


# In[7]:


hmc_sample_chains= get_hmc_n_chains(DogsModel, x_avoidance, x_shocked, y, num_chains=4, base_count = 900)


# In[8]:


# Dogs model with uniform prior
hmc_sample_chains_uniform_prior= get_hmc_n_chains(DogsModelUniformPrior, x_avoidance, x_shocked, y, num_chains=4, base_count = 900)


# ### 4. Diagnosing model fit

# **Parameter vs. Chain matrix**

# In[9]:


beta_chain_matrix_df = pd.DataFrame(hmc_sample_chains)
# beta_chain_matrix_df.to_csv("dogs_log_regression_hmc_sample_chains.csv", index=False)
beta_chain_matrix_df


# **Key statistic results as dataframe**

# In[10]:


all_metric_func_map = lambda metric, vals: {"mean":np.mean(vals), "std":np.std(vals), 
                                            "25%":np.quantile(vals, 0.25), 
                                            "50%":np.quantile(vals, 0.50), 
                                            "75%":np.quantile(vals, 0.75)}.get(metric)


# In[11]:


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

# In[12]:


fit_df = pd.DataFrame()
for chain, values in hmc_sample_chains.items():
    param_df = pd.DataFrame(values)
    param_df["chain"]= chain
    fit_df= pd.concat([fit_df, param_df], axis=0)

fit_df.to_csv("data/dogs_classification_hmc_samples.csv", index=False)    
fit_df


# In[13]:


# Use/Uncomment following once the results from pyro sampling operation are saved offline
# fit_df= pd.read_csv("data/dogs_classification_hmc_samples.csv")

# fit_df


# In[14]:


summary_stats_df_2= pd.DataFrame()

for param in ["alpha", "beta"]:
    for name, groupdf in fit_df.groupby("chain"):
        groupdi = dict(groupdf[param].describe())

        values = dict(map(lambda key:(key, [groupdi.get(key)]), ['mean', 'std', '25%', '50%', '75%']))
        values.update({"parameter": param, "chain":name})
        summary_stats_df= pd.DataFrame(values)
        summary_stats_df_2= pd.concat([summary_stats_df_2, summary_stats_df], axis=0)
summary_stats_df_2.set_index(["parameter", "chain"], inplace=True)
summary_stats_df_2


# **Following Plots m parameters side by side for n chains**

# In[15]:


parameters= ["alpha", "beta"]# All parameters for given model
chains= fit_df["chain"].unique()# Number of chains sampled for given model


func_all_params_per_chain = lambda param, chain: (param, fit_df[fit_df["chain"]==chain][param].tolist())
func_all_chains_per_param = lambda chain, param: (f'{chain}', fit_df[param][fit_df["chain"]==chain].tolist())

di_all_params_per_chain = dict(map(lambda param: func_all_params_per_chain(param, "chain_0"), parameters))
di_all_chains_per_param = dict(map(lambda chain: func_all_chains_per_param(chain, "beta"), chains))


# In[16]:


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


# In[17]:


# Use plot_interactive=True for plotly plots offline

plot_parameters_for_n_chains(chains=['chain_0', 'chain_1', 'chain_2', 'chain_3'], parameters=parameters, plot_interactive=False)


# **Joint distribution of pair of each parameter sampled values**

# In[18]:


all_combination_params = list(itertools.combinations(parameters, 2))

for param_combo in all_combination_params:
    param1, param2= param_combo
    print("\nPyro -- %s"%(f'{param1} Vs. {param2}'))
    sns.jointplot(data=fit_df, x=param1, y=param2, hue= "chain")
    plt.title(f'{param1} Vs. {param2}')
    plt.show()
    


# **Pairplot distribution of each parameter with every other parameter's sampled values**

# In[19]:


sns.pairplot(data=fit_df, hue= "chain");


# **Hexbin plots**

# In[20]:


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

def plot_interaction_hexbins(fit_df, parameters=["alpha", "beta"]):
    all_combination_params = list(itertools.combinations(parameters, 2))
    for param1, param2 in all_combination_params:#Plots interaction between each of two parameters
        hexbin_plot(fit_df[param1], fit_df[param2], param1, param2)


# In[21]:



plot_interaction_hexbins(fit_df, parameters=parameters)


# ### 5. Model evaluation: Posterior predictive checks

# **Some additional posterior analysis**
# **Pick samples from one particular chain of HMC samples**

# In[25]:


for chain, samples in hmc_sample_chains.items():
    samples= dict(map(lambda param: (param, torch.tensor(samples.get(param))), samples.keys()))# np array to tensors
    print(chain, "Sample count: ", len(samples["alpha"]))


# **Plot density for parameters from `chain_3`**

# In[26]:


title= "parameter distribution for : %s"%(chain)
fig = ff.create_distplot(list(map(lambda x:x.numpy(), samples.values())), list(samples.keys()))
fig.update_layout(title=title, xaxis_title="parameter values", yaxis_title="density", legend_title="parameters")
fig.show()

print("Alpha Q(0.5) :%s | Beta Q(0.5) :%s"%(torch.quantile(samples["alpha"], 0.5), torch.quantile(samples["beta"], 0.5)))


# **Plot density & contours for parameters from `chain_3`**

# In[27]:



fit_df = pd.DataFrame()
for chain, values in hmc_sample_chains.items():
    param_df = pd.DataFrame(values)
    param_df["chain"]= chain
    fit_df= pd.concat([fit_df, param_df], axis=0)

#Choosing samples from chain 3
chain_samples_df= fit_df[fit_df["chain"]==chain].copy()# chain is 'chain_3' 

alpha= chain_samples_df["alpha"].tolist()
beta= chain_samples_df["beta"].tolist()
colorscale = ['#7A4579', '#D56073', 'rgb(236,158,105)', (1, 1, 0.2), (0.98,0.98,0.98)]
fig = ff.create_2d_density(alpha, beta, colorscale=colorscale, hist_color='rgb(255, 255, 150)', point_size=4, title= "alpha beta joint density plot")
fig.update_layout( xaxis_title="x (alpha)", yaxis_title="y (beta)")

fig.show()


# **Note:** The distirbution of alpha values are significantly offset to the left from beta values, by almost 13 times; Thus for any given input observation of avoidances or shocked, the likelihood of getting shocked is more influenced by small measure of avoidance than getting shocked.

# **Observations:**
# 
# **Also observing the spread of alpha & beta values, the parameter beta being less negative & closer to zero can be interpreted as `learning ability`, ability of dog to learn from experiencing shocks as the increase in number of shock experiences only wavers the probability of non-avoidance (value of 𝜋𝑗) so much; So unless the trials & shocks increase considerably in progression it doesn't mellow down well and mostly stays around 0.9.**
# 
# **Whereas its not the case with alpha, alpha being more negative & farthest from zero imparts a significant decline in non-avoidance (𝜋𝑗) even for few instances where dog avoids the shock, therefore alpha can be interpreted as `retention ability`, ability to retain the learning from previous shock experiences.**

# In[36]:


print(chain_samples_df["alpha"].describe(),"\n\n", chain_samples_df["beta"].describe())


# **Minimum for alpha is `-0.225001` & beta `-0.016352`**

# In[56]:


select_sample_df= chain_samples_df[(chain_samples_df["alpha"]<-0.2)&(chain_samples_df["beta"]<-0.01)]

# print(select_sample_df.set_index(["alpha", "beta"]).index)
print("Count of alpha-beta pairs of interest, from 3rd quadrant in contour plot above (alpha<-0.2, beta <-0.01): ", select_sample_df.shape[0])

select_sample_df.head(3)


# In[61]:


select_sample_df[(select_sample_df["alpha"]<-0.22)&(select_sample_df["beta"]<-0.01)]


# **Picking a case of 3 trials with Y [0,1,1], i.e. Dog is shocked in 1st, Dogs avoids in 2nd & thereafter, effectively having an experience of 1 shock & 1 avoidance. Considering all values of alpha < `-0.2` & beta <`-0.01`**

# In[81]:



Y_li= []
Y_val_to_param_dict= defaultdict(list)

# Value alpha < -0.2 & beta < -0.01
for rec in select_sample_df.iterrows():# for beta < -0.031056
    a,b = float(rec[1]["alpha"]), float(rec[1]["beta"])
    res= round(np.exp(a+b), 4)
    Y_li.append(res)
    Y_val_to_param_dict[res].append((round(a,5),round(b,5)))# Sample-- {0.8047: [(-0.18269378, -0.034562342), (-0.18383412, -0.033494473)], 0.8027: [(-0.18709463, -0.03263992), (-0.18464606, -0.035114493)]}


# In[83]:


Y_for_select_sample_df = pd.DataFrame({"Y_for_alpha <-0.2 & beta < -0.01": Y_li})
fig = px.histogram(Y_for_select_sample_df, x= "Y_for_alpha <-0.2 & beta < -0.01")# ["Y_for_beta < -0.031056"]
title= "observed values distribution for params alpha <-0.2 & beta < -0.01"

fig.update_layout(title=title, xaxis_title="observed values", yaxis_title="count", legend_title="dogs")
fig.show()
print("Mean: %s | Median: %s"%(np.mean(Y_li), np.quantile(Y_li, 0.5)))
print("Sorted observed values: \n", sorted(Y_li))


# **For given experiment of 3 trials, from all the Ys corresponding to 38 alpha-beta pairs of interest Lowest `Y is 0.7884`; Thus selecting its corresponding alpha-beta pair**
# 
# **Note:** Can add multiple lower observed values for comparison

# In[108]:


lowest_obs = [0.7884, 0.7923, 0.7927]# Pick values from above histogram range or sorted list

selected_pairs= list(itertools.chain.from_iterable(map(lambda obs: Y_val_to_param_dict.get(obs), lowest_obs)))
selected_pairs


# **Following stores a dictionary of observed y values for pair of alpha-beta parameters**

# In[109]:


def get_obs_y_dict(select_pairs, x_a, x_s):
    """
    select_pairs: pairs of (alpha, beta) selected
    x_a: array holding avoidance count for all dogs & all trials, example for 30 dogs & 25 trials, shaped (30, 25)
    x_s: array holding shock count for all dogs & all trials, example for 30 dogs & 25 trials, shaped (30, 25)
    
    """
    y_dict = {}
    for alpha, beta in select_pairs:# pair of alpha, beta
        y_dict[(alpha, beta)] = torch.exp(alpha*x_a + beta* x_s)
    
    return y_dict


obs_y_dict= get_obs_y_dict(selected_pairs, x_avoidance, x_shocked)

print("Alpha-beta pair values as Keys to access corresponding array of inferred observations: \n", list(obs_y_dict.keys()))


# **Following plots scatterplots of observed y values for all 30 dogs for each alpha-beta pair of interest**

# In[143]:


def plot_observed_y_given_parameters(observations_list, selected_pairs_list, observed_y, chain):
    """
    observations_list:list of observated 'y' values from simulated 3 trials experiment computed corresponding 
                      to selected pairs of (alpha, beta)
    selected_pairs_list: list of alpha, beta pair tuples, example :  [(-0.225, -0.01272), (-0.21844, -0.01442)]
    
    observed_y: dict holding observed values correspodning to pair of alpha, beta tuple as key, 
                example: {(-0.225, -0.01272): tensor([[1.0000, 0.9874,..]])} 

    returns  plotly scatter plots with number of trials on X axis & corresponding probability of getting
    shocked for each pair of (alpha, beta) passed in 'selected_pairs_list'.
    
    """
    for record in zip(observations_list, selected_pairs_list):
        sim_y, select_pair = record
        print("\nFor simulated y value: %s & Selected pair: %s"%(sim_y, select_pair))
        obs_y_df= pd.DataFrame(observed_y[select_pair].numpy().T, columns=[f'Dog_{ind+1}'for ind in range(dogs_data["Ndogs"])])

        obs_y_title= "Observed values distribution for all dogs given parameter %s from %s"%(select_pair, chain)
        fig = px.scatter(obs_y_df, title=obs_y_title)
        fig.update_layout(title=obs_y_title, xaxis_title="Trials", yaxis_title="Probability of shock at trial j (𝜋𝑗)", legend_title="Dog identifier")
        fig.show()


# In[144]:


plot_observed_y_given_parameters(lowest_obs, selected_pairs, obs_y_dict, chain)


# **Following plots a single scatterplots for comparison of observed y values for all alpha-beta pairs of interest**

# In[142]:


def compare_dogs_given_parameters(pairs_to_compare, observed_y, alpha_by_beta_dict= {}):
    """
    pairs_to_compare: list of alpha, beta pair tuples to compare, 
                      example :  [(-0.225, -0.0127), (-0.218, -0.0144)]
    observed_y: dict holding observed values correspodning to pair of alpha,
                      beta tuple as key, example: {(-0.225, -0.01272): tensor([[1.0000, 0.9874,..]])} 
    alpha_by_beta_dict: holds alpha, beta pair tuples as keys & alpha/beta as value, example:
                        {(-0.2010, -0.0018): 107.08}

    returns a plotly scatter plot with number of trials on X axis & corresponding probability of getting
    shocked for each pair of (alpha, beta) passed for comparison.
    
    """
    combined_pairs_obs_df= pd.DataFrame()
    title_txt = ""
    additional_txt = ""
    for i, select_pair in enumerate(pairs_to_compare):
        i+=1
        title_txt+=f'Dog_X_m_{i} corresponds to {select_pair}, '
        if alpha_by_beta_dict:
            additional_txt+=f'𝛼/𝛽 for Dog_X_m_{i} {round(alpha_by_beta_dict.get(select_pair), 2)}, '
        
        obs_y_df= pd.DataFrame(observed_y[select_pair].numpy().T, columns=[f'Dog_{ind+1}_m_{i}'for ind in range(dogs_data["Ndogs"])])

        combined_pairs_obs_df= pd.concat([combined_pairs_obs_df, obs_y_df], axis=1)

    print(title_txt)
    print("\n%s"%additional_txt)
    obs_y_title= "Observed values for all dogs given parameter for a chain"
    fig = px.scatter(combined_pairs_obs_df, title=obs_y_title)
    fig.update_layout(title=obs_y_title, xaxis_title="Trials", yaxis_title="Probability of shock at trial j (𝜋𝑗)", legend_title="Dog identifier")
    fig.show()


# In[139]:


compare_dogs_given_parameters(selected_pairs, obs_y_dict)


# **Observations:** The 3 individual scatter plots above correspond to 3 most optimum alpha-beta pairs from 3rd quadrant of contour plot drawn earlier; Also the scatterplot following them faciliates comparing obeserved y values for all 3 pairs at once:
# 
#     1. Avoidance learning for First four dogs in the experiment favours m2 parameters (-0.21844, -0.01442), over m1 & m3 at all levels of 30 trials.
# 
#     2. learning for rest 26 dogs in the experiment favours m1 parameters (-0.225, -0.01272), over m2 & m3 at all levels of 30 trials.
# 
#     3. Learning for last four dogs 26-30 showed affinity for m3 (-0.22211, -0.01024) over m2, although even their learning was optimum with m1 parameters (-0.225, -0.01272) over rest for all trials.
# 
#     4. Although all dogs beyond Dogs 4 learned significantly better with m1 than with any other set of parameters, Dogs 8-11 learnt to avoid the shocks by trial 18, Dogs 12-16 by trial 16, All Dogs beyond number 17 learnt to dodge the shocks almost entirely within 13 trials itself.

# **Plotting observed values y corresponding to pairs of alpha-beta with with minmum & maximum value of  alpha/beta**

# **Following computes `alpha/beta` for each pair of alpha, beta and outputs pairs with maximum & minimum values; that can therefore be marked on a single scatterplots for comparison of observed y values for all alpha-beta pairs of interest**

# In[199]:


def get_alpha_by_beta_records(chain_df):
    """
    chain_df: dataframe holding sampled parameters for a given chain
    
    returns an alpha_by_beta_dictionary with alpha, beta pair tuples as keys & alpha/beta as value,
    example: {(-0.2010, -0.0018): 107.08}
    
    """
    alpha_beta_dict= {}
    
    chain_df["alpha_by_beta"] = chain_df["alpha"]/chain_df["beta"]
    min_max_values = dict(chain_df["alpha_by_beta"].describe())
    alpha_beta_list= list(map(lambda key: chain_df[chain_df["alpha_by_beta"]==min_max_values.get(key)].set_index(["alpha", "beta"])["alpha_by_beta"].to_dict(), ["max", "min"]))

    [alpha_beta_dict.update(element) for element in alpha_beta_list];
    return alpha_beta_dict


alpha_by_beta_dict = get_alpha_by_beta_records(chain_samples_df)# outputs a dict of type {(-0.2010, -0.0018): 107.08}
print("Alpha-beta pair with value as alpha/beta: ", alpha_by_beta_dict)


alpha_by_beta_selected_pairs= list(alpha_by_beta_dict.keys())
alpha_by_beta_obs_y_dict = get_obs_y_dict(alpha_by_beta_selected_pairs, x_avoidance, x_shocked)# Outputs observed_values for given (alpha, beta)


# In[222]:


compare_dogs_given_parameters(alpha_by_beta_selected_pairs, alpha_by_beta_obs_y_dict, alpha_by_beta_dict= alpha_by_beta_dict)


# **Observations:** The scatter plots above corresponds to 2 pairs of alpha-beta values from 3rd quadrant of contour plot drawn earlier, which correspond to maxmimum & minimum value of 𝛼/𝛽. Plot faciliates comparing obeserved y values for all both pairs at once:
# 
#     1. Avoidance learning for First 7 dogs in the experiment favours m2 parameters (-0.180, -0.0154) with lower 𝛼/𝛽 around 11, over m1 at all levels of 30 trials.
# 
#     2. learning for rest 23 dogs in the experiment showed affinity for m1 parameters (-0.201, -0.001) with high 𝛼/𝛽 around 107 over m2 at all levels of 30 trials, where although initial few learnt to avoid the shocks only after 20th trial, but most learnt to dodge the shocks almost entirely within 15 trials itself.

# ### 6. Model Comparison
# **Compare Dogs model with Normal prior & Uniform prior using Deviance Information Criterion (DIC)**

# **DIC is computed as follows**
# 
# $D(\alpha,\beta) = -2\ \sum_{i=1}^{n} \log P\ (y_{i}\ /\ \alpha,\beta)$
# 
# $\log P\ (y_{i}\ /\ \alpha,\beta)$ is the log likehood of shocks/avoidances observed given parameter $\alpha,\beta$, this expression expands as follows:
# 
# $$D(\alpha,\beta) = -2\ \sum_{i=1}^{30}[ y_{i}\ (\alpha Xa_{i}\ +\beta\ Xs_{i}) + \ (1-y_{i})\log\ (1\ -\ e^{(\alpha Xa_{i}\ +\beta\ Xs_{i})})]$$
# 
# 
# #### Using $D(\alpha,\beta)$ to Compute DIC
# 
# $\overline D(\alpha,\beta) = \frac{1}{T} \sum_{t=1}^{T} D(\alpha,\beta)$
# 
# $\overline \alpha = \frac{1}{T} \sum_{t=1}^{T}\alpha_{t}\\$
# $\overline \beta = \frac{1}{T} \sum_{t=1}^{T}\beta_{t}$
# 
# $D(\overline\alpha,\overline\beta) = -2\ \sum_{i=1}^{30}[ y_{i}\ (\overline\alpha Xa_{i}\ +\overline\beta\ Xs_{i}) + \ (1-y_{i})\log\ (1\ -\ e^{(\overline\alpha Xa_{i}\ +\overline\beta\ Xs_{i})})]$
# 
# 
# **Therefore finally**
# $$
# DIC\ =\ 2\ \overline D(\alpha,\beta)\ -\ D(\overline\alpha,\overline\beta)
# $$
# 
# 

# In[22]:


def calculate_deviance_given_param(parameters, x_avoidance, x_shocked, y):
    """
    D(Bt)   : Summation of log likelihood / conditional probability of output, 
              given param 'Bt' over all the 'n' cases.
    """

    D_bt_ = []
    p = parameters["alpha"]*x_avoidance + parameters["beta"]*x_shocked# alpha * Xai + beta * Xsi
    p=p.double()
    p= torch.where(p<-0.0001, p, -0.0001).float()
    
    Pij_vec = p.flatten().unsqueeze(1)# shapes (750, 1)
    Yij_vec= y.flatten().unsqueeze(0)# shapes (1, 750)
    
    # D_bt = -2 * Summation_over_i-30 (yi.(alpha.Xai + beta.Xsi)+ (1-yi).log (1- e^(alpha.Xai + beta.Xsi)))
    D_bt= torch.mm(Yij_vec, Pij_vec) + torch.mm(1-Yij_vec, torch.log(1- torch.exp(Pij_vec)))
    D_bt= -2*D_bt.squeeze().item()
    return D_bt

def calculate_mean_deviance(samples, x_avoidance, x_shocked, y):
    """
        D(Bt)_bar: Average of D(Bt) values calculated for each 
                   Bt (Bt is a single param value from chain of samples)
    """
    samples_count = list(samples.values())[0].size()[0]
    all_D_Bts= []
    for index in range(samples_count):# pair of alpha, beta
        samples_= dict(map(lambda param: (param, samples.get(param)[index]), samples.keys()))
        
        D_Bt= calculate_deviance_given_param(samples_, x_avoidance, x_shocked, y)
        all_D_Bts.append(D_Bt)
    
    D_Bt_mean = torch.mean(torch.tensor(all_D_Bts))
    
    D_Bt_mean =D_Bt_mean.squeeze().item()
    
    return D_Bt_mean
        


# In[23]:


def DIC(sample_chains, x_avoidance, x_shocked, y):
    """
        D_mean_parameters: 𝐷(𝛼_bar,𝛽_bar), Summation of log likelihood / conditional probability of output, 
                   given average of each param 𝛼, 𝛽, over 's' samples, across all the 'n' cases.
        D_Bt_mean: 𝐷(𝛼,𝛽)_bar, Summation of log likelihood / conditional probability of output, 
                   given param 𝛼, 𝛽, across all the 'n' cases.
        
        𝐷𝐼𝐶 is computed as 𝐷𝐼𝐶 = 2 𝐷(𝛼,𝛽)_bar − 𝐷(𝛼_bar,𝛽_bar)
    """
    dic_list= []
    for chain, samples in sample_chains.items():
        samples= dict(map(lambda param: (param, torch.tensor(samples.get(param))), samples.keys()))# np array to tensors

        mean_parameters = dict(map(lambda param: (param, torch.mean(samples.get(param))), samples.keys()))
        D_mean_parameters = calculate_deviance_given_param(mean_parameters, x_avoidance, x_shocked, y)

        D_Bt_mean = calculate_mean_deviance(samples, x_avoidance, x_shocked, y)
        dic = round(2* D_Bt_mean - D_mean_parameters,3)
        dic_list.append(dic)
        print(". . .DIC for %s: %s"%(chain, dic))
    print("\n. .Mean Deviance information criterion for all chains: %s\n"%(round(np.mean(dic_list), 3)))

def compare_DICs_given_model(x_avoidance, x_shocked, y, **kwargs):
    """
    kwargs: dict of type {"model_name": sample_chains_dict}
    """
    for model_name, sample_chains in kwargs.items():
        print("%s\n\nFor model : %s"%("_"*30, model_name))
        DIC(sample_chains, x_avoidance, x_shocked, y)


# In[24]:


compare_DICs_given_model(x_avoidance, x_shocked, y, Dogs_normal_prior= hmc_sample_chains, Dogs_uniform_prior= hmc_sample_chains_uniform_prior)


# _______________
