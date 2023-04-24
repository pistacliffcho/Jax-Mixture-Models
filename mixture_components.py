#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Mixture Model Classes
"""

### Imports ###
import pandas as pd
import numpy as np
from scipy.stats import norm
from scipy.stats import poisson


### Individual component Classes ###

class Normal_Col:
  """Class for representing a set of components for a Gaussian column"""
  def __init__(self, n_comps, min_sd=0.01):
    self.dist = "Normal"
    self.min_sd=min_sd
    self.n_comps = n_comps

  def m_step_one(self, w, vals):
    w_sum = np.sum(w)
    w_mean = np.sum(w * vals) / w_sum
    w_mean_sq = np.sum(w * vals**2) / w_sum
    w_sd = np.sqrt(w_mean_sq - w_mean**2)
    w_sd = np.max([w_sd, self.min_sd])
    return w_mean, w_sd

  def m_step(self, w_mat, vals):
    mean_vec = np.empty(self.n_comps) 
    sd_vec = np.empty(self.n_comps)
    for i in range(self.n_comps):
      this_mean, this_sd = self.m_step_one(w_mat[:,i], vals)
      mean_vec[i] = this_mean
      sd_vec[i] = this_sd
    self.mean_vec = mean_vec
    self.sd_vec = sd_vec
  
  def comp_log_densities(self, data_vec):
    ans = np.empty((len(data_vec), self.n_comps))
    for i in range(self.n_comps):
      ans[:, i] = norm.logpdf(data_vec,
                           self.mean_vec[i],
                           self.sd_vec[i])
    return ans
  

class Poisson_Col:
  """Class for representing a set of components for a Poisson column"""
  def __init__(self, n_comps, min_mean=0.1):
    self.dist = "Poisson"
    self.min_mean = min_mean 
    self.n_comps=n_comps

  def m_step_one(self, w, vals):
    wsum = np.sum(w * vals)
    mean = wsum/np.sum(w)
    ans = np.max([mean, self.min_mean])
    return ans

  def m_step(self, w_mat, vals):
    mean_vec = np.empty(self.n_comps)
    for i in range(self.n_comps):
      mean_vec[i] = self.m_step_one(w_mat[:,i], vals)
    self.mean_vec = mean_vec
  
  def comp_log_densities(self, data_vec):
    ans = np.empty((len(data_vec), self.n_comps))
    for i in range(self.n_comps):
      ans[:, i] = poisson.logpmf(data_vec, self.mean_vec[i])
    return ans


class Cat_Col:
  """Class for representing a set of components for a Poisson column"""
  def __init__(self, n_comps, max_int, min_p=0.001):
    self.dist = 'Categorical'
    self.max_int = max_int
    self.min_p = min_p
    self.n_comps=n_comps

  def m_step_one(self, w, vals):
    ans=np.empty(self.max_int + 1)
    for i in range(self.max_int + 1):
      is_this_group = vals == i
      this_sum = np.sum(w[is_this_group]) 
      ans[i] = this_sum
    ans = ans / np.sum(ans)
    return ans

  def m_step(self, w_mat, vals):
    self.item_probs = np.empty((self.max_int+1, self.n_comps))
    for i in range(self.n_comps):
      self.item_probs[:,i] = self.m_step_one(w_mat[:,i], vals)
    self.item_log_probs = np.log(self.item_probs)

  def comp_log_densities(self, vals):
    vals = np.array(vals, dtype=int)
    ans = np.empty((len(vals), self.n_comps))
    for i in range(self.n_comps):
      this_map = self.item_log_probs[:,i]
      these_log_probs = this_map[vals]
      ans[:, i] = these_log_probs
    return ans
