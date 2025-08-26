import numpy as np

class ThompsonSamplingBayesLRAgent():
  def __init__(self, n_arms, seed, sigma_y, beta_coeffs, prior_coeff_mu_b,  prior_coeff_mu_w, prior_coeff_sigma,
               bo_params='betas3', max_beta=-100., chosen_beta_a=1, chosen_rewardsk=50):
    '''Initialize TS BayesLR agent. bo_params can be: beta_a, betas01, betas3, etc.'''
    self.reward_list  = []              # list of collected rewards
    self.return_value = 0               # cumulative reward
    self.agent_obs0_per_draw = []       # track first obs for each draw
    self.best_action_per_draw = []      # best action per draw
    self.best_post_sample_per_draw = [] # posterior sample for best action
    self.reward_per_draw = []           # reward per draw
    self.return_per_draw = []           # cumulative return per draw
    self.n_arms = n_arms                # number of actions
    self.bo_params = bo_params          # type of BO parameters (e.g., betas3)
    self.sigma_y = sigma_y              # observation noise (σ_y)
    self.beta_coeffs = beta_coeffs      # beta coefficients for BO adjustments

    # initialize prior info for each arm
    self.prior_coeff_mu_b = {}
    self.prior_coeff_mu_w = {}
    self.prior_coeff_sigma = {}
    str_info = ''
    for arm in range(n_arms):
      self.prior_coeff_mu_b[arm]  = prior_coeff_mu_b[arm]
      self.prior_coeff_mu_w[arm]  = prior_coeff_mu_w[arm]
      self.prior_coeff_sigma[arm] = prior_coeff_sigma[arm]
      str_info += '\nμb[{}]={}  μw[{}]={}'.format(arm, float(self.prior_coeff_mu_b[arm]), arm, self.prior_coeff_mu_w[arm])
    
    self.max_beta = max_beta
    self.chosen_beta_a = int(chosen_beta_a)
    self.chosen_rewardsk = int(chosen_rewardsk)
    self.seed = seed
    self.rng = np.random.default_rng(self.seed)  # RNG for sampling
    
    # agent configuration string
    self.config = 'TS BayesLR σy={} β={} prior coeffs:{} seed={} max_beta={}'.format(
                   self.sigma_y, self.beta_coeffs, str_info, self.seed, self.max_beta)
    if self.bo_params == 'beta_a': self.config += ' (beta_a={})'.format(chosen_beta_a)
    if self.bo_params == 'betas01': self.config += ' (betas01)'
    if self.bo_params == 'betas3': self.config += ' (betas3)'
    if self.bo_params == 'betas3 rewardsk': self.config += ' (betas3 rewardsk={})'.format(chosen_rewardsk)
    
    # initialize posterior dictionaries
    self.post_coeff_mu_dict = {}; self.post_coeff_sigma_dict = {}
    for arm in range(self.n_arms):
      array1 = self.prior_coeff_mu_b[arm]
      array2 = self.prior_coeff_mu_w[arm]
      self.post_coeff_mu_dict[arm] = np.concatenate((array1, array2), axis=None)  # posterior mean
      self.post_coeff_sigma_dict[arm] = np.array(self.prior_coeff_sigma[arm])     # posterior cov
    
    # store copies of prior info
    self.prior_coeff_mu_info_dict = self.post_coeff_mu_dict.copy()
    self.prior_coeff_sigma_info_dict = self.post_coeff_sigma_dict.copy()
    
    # track previous posterior (default: arm 0)
    self.previous_post_coeff_mu    = np.copy(self.post_coeff_mu_dict[0])
    self.previous_post_coeff_sigma = np.copy(self.post_coeff_sigma_dict[0])

  def choose_action(self, obs, env_trajectory=None):
    agent_obs_list = list(obs)
    self.agent_obs_with_intercept = np.array([1] + agent_obs_list)  # add intercept
    obs_coeff_samples = []
    
    # sample posterior coefficients and compute predicted reward
    for arm in range(self.n_arms):
      coeff_sample = self.rng.multivariate_normal(mean=self.post_coeff_mu_dict[arm], cov=self.post_coeff_sigma_dict[arm])
      obs_coeff_sample = self.agent_obs_with_intercept.T.dot(coeff_sample)
      
      # add BO parameter adjustments if needed
      if self.bo_params.find('betas3') >= 0:
        if arm > 0:
          obs_coeff_sample += self.beta_coeffs[arm-1].item() * self.max_beta
      elif self.bo_params.find('beta') < 0:
        pass  # no adjustment
      else:
        str_error_message = 'cannot find bo_params = {} in TS agent!'.format(self.bo_params)
        print('\n{}\n'.format(str_error_message))
        assert(1==2), str_error_message
      
      obs_coeff_samples.append(obs_coeff_sample)
    
    chosen_action = int(np.argmax(obs_coeff_samples))  # select action with max sampled reward
    self.agent_obs0_per_draw.append(agent_obs_list[0])
    self.best_action_per_draw.append(chosen_action)
    self.best_post_sample_per_draw.append(obs_coeff_samples[chosen_action])
    return chosen_action

  def store_rewards(self, reward):
    self.reward_list.append(reward)     # store reward
    self.return_value += reward         # update cumulative return
    self.reward_per_draw.append(reward)
    self.return_per_draw.append(self.return_value)

  def calculate_posterior_mean_sigma(self, sigma_y, prior_mu, prior_sigma_inv, X, Y):  
    # Bayesian update for mean & covariance
    post_sigma = (sigma_y**2) * np.linalg.inv( X.T @ X + (sigma_y**2) * prior_sigma_inv )
    post_mu    = post_sigma @ ( (X.T @ Y)/(sigma_y**2) + prior_sigma_inv @ prior_mu )
    return post_mu, post_sigma

  def update_posterior(self, chosen_action, reward):
    # update posterior after observing reward
    self.previous_post_coeff_mu        = self.post_coeff_mu_dict[chosen_action]
    self.previous_post_coeff_sigma     = self.post_coeff_sigma_dict[chosen_action]
    self.previous_post_coeff_sigma_inv = np.linalg.inv(self.previous_post_coeff_sigma)
    
    X = np.array([self.agent_obs_with_intercept])
    Y = np.array([reward])
    
    chosen_sigma_y = np.array(self.sigma_y[chosen_action]) if self.bo_params.find('sigmaYs4') >= 0 else self.sigma_y
    
    # Bayesian posterior update
    self.post_coeff_mu_dict[chosen_action], self.post_coeff_sigma_dict[chosen_action] = self.calculate_posterior_mean_sigma(
        chosen_sigma_y, self.previous_post_coeff_mu, self.previous_post_coeff_sigma_inv, X, Y)
