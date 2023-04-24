#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 21 21:52:49 2023

@author: cliffordanderson-bergman
"""

import unittest 
from mixture_components import *

vals = np.array([1, 2, 3])
w_mat = np.array([[0.1, .8, 0.1],
                  [.4, .2, .4]])
w_mat = np.transpose(w_mat)

class TestNormal(unittest.TestCase):
  @classmethod
  def setUpClass(self):
    self.norm_col = Normal_Col(n_comps=2)
    
  def test_wmle_one(self):
    res = self.norm_col.m_step_one(w=w_mat[:,0], vals=vals)
    assert res[0] == 2
    assert len(res) == 2
    
  def test_wmle(self):

unittest.main()