#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 23 09:39:05 2023

@author: cliffordanderson-bergman
"""

import jax.numpy as jnp
from jax.numpy import newaxis as nax
from jax import jit


### Jax based density functions ###
def bc_norm_lpdf(row_x_vec, col_mus, col_sds):
    """Broadcast log density of univariate normal."""
    diff = row_x_vec[:,nax] - col_mus[nax,:]
    ans = -(diff**2) / (col_sds[nax,:]**2. * 2.) -jnp.log(col_sds[nax])
    return ans

jit_norm_lpdf = jit(bc_norm_lpdf)

def bc_pois_lpmf(row_x_vec, mu_vec):
    """Broadcast log probability of poisson."""
    term_1 = row_x_vec[:,nax] * jnp.log(mu_vec[nax,:])
    ans = term_1 - mu_vec[nax,:]
    return ans

jit_pois_lpmf = jit(bc_pois_lpmf)

def bc_cat_lpmf(row_x_vec, log_p_mat):
    """Broadcast log probability of categorical.
    
    log_p_mat should be matrix where each _column_ 
    is the vector of log probabilities of a given component."""
    return log_p_mat[row_x_vec[:,nax]].squeeze()

jit_cat_lpmf = jit(bc_cat_lpmf)

### Jax based weighted mle's ###
def bc_pois_wmle(w_mat, x_vec, min=1e-4):
    """Broadcast weighted MLE of poisson distribution."""
    prod_mat = w_mat * x_vec[:,nax]
    ans = prod_mat.sum(axis=0) / w_mat.sum(axis=0)
    ans = jnp.clip(ans, a_min=min)
    return ans

jit_pois_wmle = jit(bc_pois_wmle)

def bc_norm_wmle(w_mat, x_vec, min=1e-4):
    """Broadcast weighted MLE of univariate normal distribution."""
    w_sums = w_mat.sum(axis=0)
    prod_mat = w_mat * x_vec[:,nax]
    prod2_mat = w_mat * (x_vec**2)[:,nax]
    mu_vec = prod_mat.sum(axis=0) / w_sums 
    mu2_vec = prod2_mat.sum(axis=0) / w_sums 
    sd_vec = jnp.sqrt(mu2_vec - mu_vec**2)
    sd_vec = jnp.clip(sd_vec, a_min=min)
    return (mu_vec, sd_vec)

jit_norm_wmle = jit(bc_norm_wmle)

def bc_cat_wmle(w_mat, x_vec, n_outcomes, min=1e-6):
    """Broadcast weighted MLE of univariate categorical distribution."""
    # Expanding x_vector into a new group for each component
    expanded_x_vec = x_vec[:,nax]
    expanded_x_vec = expanded_x_vec + jnp.array(range(w_mat.shape[1]))[nax,:] * n_outcomes
    bincount_res = jnp.bincount(
        expanded_x_vec.reshape(-1), 
        weights=w_mat.reshape(-1))
    tr_shape = (w_mat.shape[1], 
                   n_outcomes)
    bincount_res = bincount_res.reshape(tr_shape).transpose()
    bincount_res = jnp.clip(bincount_res, a_min=min)
    ans = bincount_res / bincount_res.sum(axis=0)[nax, :]
    return ans

class Normal_Col:
    def __init__(self, n_comps, training_vec, min_s):
        self.dist = 'Normal'
        self.n_comps = n_comps 
        self.training_vec = training_vec
        self.min_s = min_s

    def m_step(self, w_mat, new_vals=None):
        if new_vals is None:
            vals = self.training_vec
        else:
            vals = new_vals
        mean_vec, sd_vec = jit_norm_wmle(w_mat, vals, min=self.min_s)
        self.mean_vec = mean_vec
        self.sd_vec = sd_vec 

    def comp_log_densities(self, new_vals=None):
        if new_vals is None:
            vals = self.training_vec
        else:
            vals = new_vals
        return jit_norm_lpdf(vals, 
                             self.mean_vec,
                             self.sd_vec)
    
class Poisson_Col:
    def __init__(self, n_comps, training_vec, min_mu):
        self.dist = 'Poisson'
        self.n_comps = n_comps
        self.training_vec = training_vec
        self.min_mu=min_mu

    def m_step(self, w_mat, new_vals=None):
        if new_vals is None:
            vals = self.training_vec
        else:
            vals = new_vals
        self.mean_vec = jit_pois_wmle(w_mat, vals, min=self.min_mu)
    
    def comp_log_densities(self, new_vals=None):
        if new_vals is None:
            vals = self.training_vec
        else:
            vals = new_vals
        return jit_pois_lpmf(vals, 
                             self.mean_vec)
    
class Cat_Col:
    def __init__(self, n_comps, max_int, training_vec, min_p):
        self.dist = 'Categorical'
        self.n_outcomes = max_int + 1
        self.n_comps=n_comps 
        self.training_vec=training_vec
        self.min_p=min_p
    
    def m_step(self, w_mat, new_vals=None):
        if new_vals is None:
            vals = self.training_vec
        else:
            vals = new_vals
        item_probs = bc_cat_wmle(w_mat, vals, n_outcomes=self.n_outcomes, min=self.min_p)
        self.log_item_probs = jnp.log(item_probs)

    def comp_log_densities(self, new_vals=None):
        if new_vals is None:
            vals = self.training_vec
        else:
            vals = new_vals
        return jit_cat_lpmf(vals, 
                           self.log_item_probs)