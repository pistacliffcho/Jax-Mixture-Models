from mixture_model import * 
import jax.numpy as jnp
import numpy as np
import pandas as pd
import unittest

def create_problem(mu=0, lam=10, p=[0.3, .3, .4], size=100):
    ans = pd.DataFrame(dict(
        a=np.random.normal(loc=mu, size=size), 
        b=np.random.poisson(lam=lam, size=size),
        c=np.random.choice([0, 1, 2], p=p, size=size)
    ))
    return ans

class TestMixture(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.df = create_problem()
        self.data_mat = jnp.array(self.df)
        self.data_types = ['normal', 'poisson', 'cat']

        np.random.seed(seed=123)

    def test_builds(self):
        self.mod = MixtureModel(self.df, self.data_types, n_comps=2)
        assert isinstance(self.mod, MixtureModel)

    def test_llk_improves(self):
        mod = MixtureModel(self.df, 
                                self.data_types,
                                n_comps=2)
        llk_1 = mod.llk() 
        mod.em_step() 
        llk_2 = mod.llk()
        assert llk_2 > llk_1

    def test_finds_reasonable_probalities(self):
        df1 = create_problem(size=500)
        df2 = create_problem(size=2000, 
                             mu=2.5, 
                             lam=15,
                             p=[.8, .1, .1])
        n1 = df1.shape[0]
        n2 = df2.shape[0]
        n_tot = n1 + n2
        p_true = [np.min([n1, n2])/n_tot, 
                  np.max([n1, n2])/n_tot]
        df_both = pd.concat([df1, df2], ignore_index=True)
        mod = MixtureModel(df_both,
                           data_types=self.data_types, 
                           n_comps=2)
        mod.fit()
        p_est = np.sort(mod.p_vec)
        np.testing.assert_allclose(p_est, p_true, atol=1e-2)

unittest.main()