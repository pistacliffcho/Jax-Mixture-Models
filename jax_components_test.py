from jax_components import * 
from scipy.stats import norm
from scipy.stats import poisson
import numpy as np
import unittest

class TestDensities(unittest.TestCase):
    def test_normal(self):
        # Sample data
        x_vec = jnp.array([1,2, 3])
        mu_vec = jnp.array([4,3])
        sd_vec = jnp.array([1, 0.2])
        # Computing jax answer
        jax_answer = bc_norm_lpdf(x_vec, mu_vec, sd_vec)
        # Manually evaluation
        manual_ans = jnp.zeros([len(x_vec),len(mu_vec)])
        for i in range(len(x_vec)):
            for j in range(len(mu_vec)):
                this_val = norm.logpdf(x_vec[i], mu_vec[j], sd_vec[j])
                manual_ans = manual_ans.at[i,j].set(this_val)
        # Answer should be correct up to an additive constant for fixed x
        # Subtracting row means should set equal 
        jax_answer = jax_answer - jnp.mean(jax_answer, axis=1)[:,nax]
        manual_ans = manual_ans - jnp.mean(manual_ans, axis=1)[:,nax]
        np.testing.assert_allclose(jax_answer, manual_ans, rtol=10**-4)

    def test_poisson(self):
        # Sample data
        x_vec = jnp.array([1,2,3])
        mu_vec = jnp.array([4,3])
        # Jax answer
        jax_answer = bc_pois_lpmf(x_vec, mu_vec)
        # Manually evaluation
        manual_ans = jnp.zeros([len(x_vec),
                                len(mu_vec)])
        for i in range(len(x_vec)):
            for j in range(len(mu_vec)):
                this_val = poisson.logpmf(x_vec[i], mu_vec[j])
                manual_ans = manual_ans.at[i,j].set(this_val)

        jax_answer = jax_answer - jnp.mean(jax_answer, axis=1)[:,nax]
        manual_ans = manual_ans - jnp.mean(manual_ans, axis=1)[:,nax]
        np.testing.assert_allclose(jax_answer, manual_ans, rtol=10**-4)

    def test_cat(self):
        # Making data
        x_vec = jnp.array([0, 1, 2, 1])
        log_p_mat = jnp.log(
            jnp.array([[0.1, 0.8, 0.1], 
             [.2, .4, .4]])
        )
        # Jax answer
        jax_answer = bc_cat_lpmf(x_vec, log_p_mat)
        # Manual checking of answer
        manual_ans = jnp.zeros([len(x_vec), 
                                log_p_mat.shape[1]])
        for i in range(len(x_vec)):
            for j in range(log_p_mat.shape[1]):
                this_ans = log_p_mat[x_vec[i], j]
                manual_ans = manual_ans.at[i,j].set(this_ans)
        np.testing.assert_allclose(jax_answer, manual_ans, rtol=10**-4)


class Testwmles(unittest.TestCase):
    def test_pois(self):
        # Data
        x_vec = jnp.array([1,2,3])
        w_mat = jnp.array([[0.2, .3, .7],
                           [.1, .1, .8]]).transpose()
        # Jax answer
        jax_answer = bc_pois_wmle(w_mat, x_vec)
        # Manual answer
        manual_answer = jnp.zeros(w_mat.shape[1])
        for j in range(w_mat.shape[1]):
            this_ans = jnp.sum(x_vec * w_mat[:, j]) / jnp.sum(w_mat[:,j])
            manual_answer = manual_answer.at[j].set(this_ans)
        np.testing.assert_allclose(jax_answer, manual_answer, rtol=1e-4)
        
    def test_normal(self):
        # Data
        x_vec = jnp.array([1,2,3])
        w_mat = jnp.array([[0.2, .3, .7],
                           [.1, .1, .8]]).transpose()
        # Jax answer
        jax_mean, jax_sd = bc_norm_wmle(w_mat, x_vec)
        manual_mean = []
        manual_sd = [] 
        for j in range(w_mat.shape[1]):
            this_wsum = jnp.sum(w_mat[:,j])
            this_mean = jnp.sum(x_vec * w_mat[:,j]) / this_wsum 
            this_var = jnp.sum(x_vec**2 * w_mat[:,j])/this_wsum - this_mean**2 
            this_sd = jnp.sqrt(this_var)
            manual_mean.append(this_mean)
            manual_sd.append(this_sd)
        manual_mean = jnp.array(manual_mean)
        manual_sd = jnp.array(manual_sd)
        np.testing.assert_allclose(
            jax_mean, manual_mean, rtol=1e-4
        )
        np.testing.assert_allclose(
            jax_sd, manual_sd, rtol=1e-4
        )

    def test_cat(self):
        # Data
        x_vec = jnp.array([1,2,2,0,1])
        n_outcomes = jnp.max(x_vec) + 1
        w_mat = jnp.array([[0.2, .3, .5, .1, .1],
                           [.1, .1, .7, .05, .05]]).transpose()
        # jax result
        jax_results = bc_cat_wmle(w_mat, x_vec, n_outcomes=n_outcomes)

        # manual_results
        manual_results = jnp.zeros((n_outcomes,
                                    w_mat.shape[1]))
        for i in range(n_outcomes):
            is_this_group = x_vec == i
            for j in range(w_mat.shape[1]):
                this_sum = jnp.sum(w_mat[is_this_group,j])
                manual_results = manual_results.at[i,j].set(this_sum)
        for j in range(w_mat.shape[1]):
            manual_results = manual_results.at[:,j].set(manual_results[:,j] / jnp.sum(manual_results[:,j]))
        np.testing.assert_allclose(jax_results, manual_results)

class TestJIT(unittest.TestCase):
    def test_densties(self):
        # Sample data
        x_vec = jnp.array([1,2, 3])
        mu_vec = jnp.array([4,3])
        sd_vec = jnp.array([1, 0.2])

        # Normal
        raw_ans = bc_norm_lpdf(x_vec, mu_vec, sd_vec)
        jit_ans = jit_norm_lpdf(x_vec, mu_vec, sd_vec)
        np.testing.assert_allclose(raw_ans, jit_ans, rtol=1e-4)

        # Poisson
        raw_ans = bc_pois_lpmf(x_vec, mu_vec)
        jit_ans = jit_pois_lpmf(x_vec, mu_vec)
        np.testing.assert_allclose(raw_ans, jit_ans, rtol=1e-4)

        # Categorical 
        p_mat = jnp.array(([0.2, 0.2, .6], 
                           [.1, .1, .8])).transpose()
        log_p_mat = jnp.log(p_mat)
        raw_ans = bc_cat_lpmf(x_vec, 
                              log_p_mat)
        jit_ans = jit_cat_lpmf(x_vec, log_p_mat)
        np.testing.assert_allclose(raw_ans, jit_ans, rtol=1e-4)

    def test_wmles(self):
        # data
        x_vec = jnp.array([1,2,3])
        w_mat = jnp.array([[0.2, .3, .7],
                           [.1, .1, .8]]).transpose()

        raw_ans = bc_norm_wmle(w_mat, x_vec)
        jit_ans = bc_norm_wmle(w_mat, x_vec)
        np.testing.assert_allclose(raw_ans, jit_ans, rtol=1e-4)

        raw_ans = bc_pois_wmle(w_mat, x_vec)
        jit_ans = jit_pois_wmle(w_mat, x_vec)
        np.testing.assert_allclose(raw_ans, jit_ans, rtol=1e-4)

        # Note: jit compiling of categorical does not work due to dynamic indices
        # n_outcomes = jnp.max(x_vec) + 1
        # raw_ans = bc_cat_wmle(w_mat, x_vec, n_outcomes)
        # jit_ans = jit_cat_wmle(w_mat, x_vec, n_outcomes)
        # np.testing.assert_allclose(raw_ans, jit_ans, rtol=1e-4)

unittest.main()