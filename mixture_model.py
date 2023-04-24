from jax_components import *
import numpy as np

class MixtureModel:
  def __init__(self, df, data_types, n_comps=2):
    assert len(data_types) == df.shape[1]
    self.training_df = df
    self.n_comps = n_comps
    self.n_rows = self.training_df.shape[0]
    self.n_cols = self.training_df.shape[1]
    self.comp_vars = {}
    for i, this_col in enumerate(df.columns):
      if data_types[i] == "normal":
        this_vec = jnp.array(df[this_col], dtype=jnp.float16)
        min_s = jnp.std(this_vec) / (10 * n_comps)
        self.comp_vars[this_col] = Normal_Col(n_comps=n_comps,
                                              training_vec=this_vec,
                                              min_s=min_s)
      elif data_types[i] == 'cat':
        this_vec = np.array(df[this_col], dtype=jnp.int16)
        max_int = jnp.max(this_vec)
        min_p = 1 / (10 * len(this_vec))
        self.comp_vars[this_col] = Cat_Col(n_comps=n_comps,
                                           max_int=max_int, 
                                           training_vec=this_vec,
                                           min_p=min_p)
      elif data_types[i] == 'poisson':
        this_vec = np.array(df[this_col], dtype=jnp.int16)
        min_mu = 1e-8
        self.comp_vars[this_col] = Poisson_Col(n_comps=n_comps,
                                               training_vec=this_vec,
                                               min_mu=min_mu)
      else:
        raise ValueError("data_types " + data_types[i] + " not recognize")

    # Random initialization
    self.p_mat = jnp.array(
      np.random.dirichlet(
        alpha = [1] * self.n_comps, 
        size=self.n_rows))
    self.p_vec = jnp.mean(self.p_mat, axis=0)
    self.em_step()
    
  def get_comp_log_densities(self, new_data=None):
    ans = jnp.zeros((self.n_rows, self.n_comps))
    for i, k in enumerate(self.comp_vars.keys()):
        ans+= self.comp_vars[k].comp_log_densities(new_vals=new_data)
    return ans

  def make_p_mat(self, new_data=None):
    p_mat = jnp.exp(self.get_comp_log_densities(new_data)) * self.p_vec[jnp.newaxis,:]
    row_sums = jnp.sum(p_mat, axis=1)
    p_mat /= row_sums[:, jnp.newaxis]
    return p_mat

  def em_step(self):
    # M-step
    for i, k in enumerate(self.comp_vars.keys()):
      self.comp_vars[k].m_step(self.p_mat)
    # E-step 
    self.p_mat = self.make_p_mat()
    self.p_vec = jnp.mean(self.p_mat, axis=0)
  
  def llk(self):
    dens_mat = jnp.zeros((self.n_rows, self.n_comps))
    dens_mat = dens_mat + self.p_vec[nax,:]
    dens_mat *= jnp.exp(self.get_comp_log_densities())
    lk_vec = jnp.sum(dens_mat, axis=1)
    llk_vec = jnp.log(lk_vec)
    return(jnp.sum(llk_vec))
 
  def fit(self, max_iter=1000, tol = 0.01, llk_check_period=10, verbose=False):
    llk = self.llk() 
    if verbose:
      print("llk: " + str(llk))
    conv = False
    for i in range(max_iter):
      self.em_step() 
      if i % llk_check_period == (llk_check_period - 1):
        new_llk = self.llk() 
        err = new_llk - llk 
        llk = new_llk
        if verbose:
          print("llk: " + str(llk))
        if err < tol:
          conv = True
          break 
    ans = dict(err=err, conv=conv, iters = i)
    return ans
