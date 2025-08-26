import math
import torch
import gpytorch
import numpy as np
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.constraints import Interval
from gpytorch.kernels import MaternKernel, ScaleKernel
from botorch.acquisition import qExpectedImprovement
from botorch.generation import MaxPosteriorSampling
from botorch.utils.transforms import normalize, unnormalize
from botorch import fit_gpytorch_mll
from botorch.optim import optimize_acqf
from botorch.models import SingleTaskGP
from torch.quasirandom import SobolEngine
from dataclasses import dataclass
from abc import ABC, abstractmethod
import torch.nn.functional as F

from BOTS.agents import ThompsonSamplingBayesLRAgent
from BOTS.utils import get_TS_coeffs

class BatchAgentBase(ABC):
  """Abstract base class for batch RL agents."""
  def __init__(self, agent_name, agent_seed, B_schedule, agent_settings):
    """Initialize batch agent base."""
    self.info_dict = {}
    self.agent_seed = agent_seed
    self.agent_name = agent_name
    self.B_schedule = B_schedule
    self.agent_settings = agent_settings
    torch.manual_seed(agent_seed[0][0])

  @abstractmethod
  def choose_action(self, obs, current_batch):
    """Return chosen action for current batch."""
    chosen_action = None
    return chosen_action

  @abstractmethod
  def update_batch(self, b_train, info_dict, obs, action, reward, obs_, done, current_batch):
    pass

  @abstractmethod
  def update_step(self, b_train, current_step):
    pass

  @abstractmethod
  def update_round(self, b_train, info_dict, max_episode_length, current_round):
    pass

#-----------------------------------------------------------------

def create_TS_and_set_priors(agent_name, agent_seed, agent_settings, info_dict):
  """Create Thompson Sampling agent and set priors."""
  n_arms = agent_settings['n_actions']
  input_dims = len(agent_settings['chosen_obs_names'])
  default_sigma_y = agent_settings['sigma_y']
  default_prior_coeff_mu_b  = agent_settings['coeff_mu_b']
  default_prior_coeff_mu_w  = np.ones(input_dims)*agent_settings['coeff_mu_w']
  default_prior_coeff_sigma = np.eye(input_dims+1)*agent_settings['coeff_sigma']
  prior_coeff_mu_b = {}; prior_coeff_mu_w = {}; prior_coeff_sigma = {};
  for arm in range(n_arms):
    prior_coeff_mu_b[arm]  = default_prior_coeff_mu_b
    prior_coeff_mu_w[arm]  = default_prior_coeff_mu_w
    prior_coeff_sigma[arm] = default_prior_coeff_sigma
  chosen_agent = ThompsonSamplingBayesLRAgent( n_arms=n_arms, seed=agent_seed, sigma_y=default_sigma_y, beta_coeffs=agent_settings['beta_coeffs'],
                                               prior_coeff_mu_b=prior_coeff_mu_b, prior_coeff_mu_w=prior_coeff_mu_w, prior_coeff_sigma=prior_coeff_sigma,
                                               bo_params=agent_settings['bo_params'], max_beta=agent_settings['max_beta'])
  return chosen_agent

#-----------------------------------------------------------------

class BatchAgentTS(BatchAgentBase):
  """Batch agent using Thompson Sampling."""
  def __init__(self, agent_name, agent_seed, B_schedule, agent_settings):
    """Initialize TS batch agent and create initial agents."""
    self.info_dict = {}
    self.agent_seed = agent_seed
    self.agent_name = agent_name
    self.B_schedule = B_schedule
    torch.manual_seed(self.agent_seed[0][0])
    self.agent_settings = agent_settings
    self.agents = []
    if ('n_actions' not in agent_settings) or ('coeff_mu_w' not in agent_settings):
      error_message = 'Cannot find key for BatchAgentTS. BatchAgentTS agent_settings keys must contain:\n'
      error_message += 'n_actions, chosen_obs_names, sigma_y, coeff_mu_w, coeff_mu_b, coeff_sigma.'
      print(error_message)
      assert(1==2), error_message

    # create TS agent at round=0
    chosen_new_batch_size = max(self.B_schedule)
    for batch_index in range(chosen_new_batch_size):
      agent = create_TS_and_set_priors(self.agent_name, self.agent_seed[0][batch_index], self.agent_settings, self.info_dict)
      self.agents.append(agent)

  def choose_action(self, obs, current_batch):
    chosen_action = self.agents[current_batch].choose_action(obs)
    return chosen_action

  def update_batch(self, b_train, info_dict, obs, action, reward, obs_, done, current_batch):
    self.agents[current_batch].update_posterior(action, reward)

  def update_step(self, b_train, current_step):
    pass

  def update_round(self, b_train, info_dict, max_episode_length, current_round):
    """Update TS agents at each BO round if using TS_prior."""
    if self.agent_name == 'TS_prior':
      max_rounds_index = info_dict['ROUNDS']-1
      if current_round < max_rounds_index:
        chosen_new_batch_size = self.B_schedule[current_round+1]    
        self.agents = []
        for batch_index in range(chosen_new_batch_size):
          agent = create_TS_and_set_priors(self.agent_name, self.agent_seed[current_round][batch_index], self.agent_settings, self.info_dict)
          self.agents.append(agent)

#-----------------------------------------------------------------
# The TurboState class is based on the official BoTorch implementation, with modifications for BOTS.

@dataclass
class TurboState:
  """State for TuRBO trust region."""
  chosen_x_dim: int
  batch_size: int
  length: float = 0.8
  length_min: float = 0.5 ** 7
  length_max: float = 1.6
  failure_counter: int = 0
  failure_tolerance: int = float("nan")
  success_counter: int = 0
  success_tolerance: int = 10  # note: original paper uses 3
  best_value: float = -float("inf")
  restart_triggered: bool = False

def __post_init__(self):
  self.failure_tolerance = math.ceil(max([4.0 / self.batch_size, float(self.chosen_x_dim) / self.batch_size]))

def update_TurboState(input_state, y_new):
  """Update TuRBO state based on new function values."""
  if max(y_new) > input_state.best_value + 1e-3 * math.fabs(input_state.best_value):
    input_state.success_counter += 1
    input_state.failure_counter = 0
  else:
    input_state.success_counter = 0
    input_state.failure_counter += 1

  if input_state.success_counter == input_state.success_tolerance:    # expand trust region
    input_state.length = min(2.0 * input_state.length, input_state.length_max)
    input_state.success_counter = 0
  elif input_state.failure_counter == input_state.failure_tolerance:  # shrink trust region
    input_state.length /= 2.0
    input_state.failure_counter = 0

  input_state.best_value = max(input_state.best_value, max(y_new).item())
  if input_state.length < input_state.length_min:
    input_state.restart_triggered = True
  return input_state

#-------------------------------------------------------------------------

def generate_TuRBO_batch(
                          turbo_state,
                          model,    # GP model
                          X,        # evaluated points on the domain [0, 1]^d
                          Y,        # function values
                          batch_size, seed_for_TS, num_restarts, raw_samples, max_cholesky_size,
                          dtype_for_TS, device_for_TS, 
                          turbo_acq_fn_name,          # 'qEI' or 'TS'
                          n_candidates_for_TS=None,   # number of candidates for Thompson sampling (for turbo_acq_fn_name='TS')
                        ):
  """Generate candidate batch points for TuRBO using TS or qEI."""
  if n_candidates_for_TS is None:
    n_candidates_for_TS = min(5000, max(2000, 200 * X.shape[-1]))    # default is min(5000, max(2000, 200 * x_dim))

  # scale the trust region (TR) to be proportional to the GP lengthscales
  x_center = X[Y.argmax(), :].clone()
  weights = model.covar_module.base_kernel.lengthscale.squeeze().detach() 

  if len(weights.shape) < 1:
    weights = weights.unsqueeze(-1)  
  weights = weights / weights.mean()
  weights = weights / torch.prod(weights.pow(1.0 / len(weights)))
  tr_lb = torch.clamp(x_center - weights * turbo_state.length / 2.0, 0.0, 1.0)
  tr_ub = torch.clamp(x_center + weights * turbo_state.length / 2.0, 0.0, 1.0)

  if turbo_acq_fn_name == 'TS':
    chosen_dim = X.shape[-1]
    sobol = SobolEngine(chosen_dim, scramble=True, seed=seed_for_TS)    # original 100*seed_for_TS
    perturb = sobol.draw(n_candidates_for_TS).to(dtype=dtype_for_TS, device=device_for_TS)
    perturb = tr_lb + (tr_ub - tr_lb) * perturb

    # create a perturbation mask
    prob_perturb = min(20.0 / chosen_dim, 1.0)
    mask = (torch.rand(n_candidates_for_TS, chosen_dim, dtype=dtype_for_TS, device=device_for_TS) <= prob_perturb)
    ind = torch.where(mask.sum(dim=1) == 0)[0]
    mask[ind, torch.randint(0, chosen_dim - 1, size=(len(ind),), device=device_for_TS)] = 1

    # create candidate points from the perturbations and the mask
    X_cand = x_center.expand(n_candidates_for_TS, chosen_dim).clone()
    X_cand[mask] = perturb[mask]

    # sample on the candidate points
    thompson_sampling = MaxPosteriorSampling(model=model, replacement=False)
    with torch.no_grad():  # we don't need gradients when using TS
      x_new_tr = thompson_sampling(X_cand, num_samples=batch_size)

  elif turbo_acq_fn_name == 'qEI':
    with gpytorch.settings.max_cholesky_size(max_cholesky_size):
      qEI_turbo = qExpectedImprovement(model, Y.max())
      x_new_tr, acq_value = optimize_acqf(qEI_turbo, bounds=torch.stack([tr_lb, tr_ub]), q=batch_size, num_restarts=num_restarts, raw_samples=raw_samples)
  return x_new_tr

#-------------------------------------------------------------------------

class BatchAgentBO(BatchAgentBase):
  """Batch agent using Bayesian Optimization."""
  def __init__(self, agent_name, agent_seed, B_schedule, agent_settings, n_candidates_for_TS, first_round_BO=1):
    """Initialize BO batch agent and create initial TS agents."""
    self.info_dict = {}
    self.agent_seed = agent_seed
    self.agent_name = agent_name
    self.B_schedule = B_schedule
    torch.manual_seed(self.agent_seed[0][0])
    self.agent_settings = agent_settings
    self.first_round_BO = first_round_BO
    self.current_trial = self.agent_settings['trial']
    self.Y = None
    self.n_candidates_for_TS = n_candidates_for_TS # only used for BO(qTS) and TuRBO(qTS)
    self.dtype = self.agent_settings['dtype'] 
    self.device = self.agent_settings['device']
    self.default_TS_settings = self.agent_settings['default_TS_settings'].copy()
    x_dim = self.agent_settings['x_dim']

    if self.agent_name == 'Sobol':
      n_Sobol = max(2,int(np.sum(B_schedule[self.first_round_BO+1:])))
      sobol_init_seed = int(self.agent_seed[0][1])   
      self.X_Sobol_norm = SobolEngine(x_dim, scramble=True, seed=sobol_init_seed).draw(n_Sobol).to(dtype=self.dtype, device=self.device)
      self.Sobol_count_so_far = 0
    
    if self.agent_name.find('TuRBO') >= 0:
      self.turbo_state = TurboState(chosen_x_dim=x_dim, batch_size=self.B_schedule[-1])
    
    # get initial new X using Sobol
    n_init = self.B_schedule[self.first_round_BO]
    n_Sobol = max(2,n_init)
    sobol_init_seed = int(self.agent_seed[0][0])        
    self.X = SobolEngine(x_dim, scramble=True, seed=sobol_init_seed).draw(n_Sobol).to(dtype=self.dtype, device=self.device)

    # create new agents using initial new X
    chosen_new_batch_size = n_init
    self.agents = []
    for b in range(chosen_new_batch_size):
      TS_agent_settings = self.default_TS_settings.copy()
      TS_agent_settings['beta_coeffs'] = self.X[b]
      agent = create_TS_and_set_priors(self.agent_name, self.agent_seed[0][b], TS_agent_settings, self.info_dict)
      self.agents.append(agent)
    self.rewards = torch.zeros((chosen_new_batch_size), dtype=self.dtype, device=self.device)

  def choose_action(self, obs, current_batch):
    chosen_action = self.agents[current_batch].choose_action(obs)
    return chosen_action

  def update_batch(self, b_train, info_dict, obs, action, reward, obs_, done, current_batch):
    self.agents[current_batch].update_posterior(action, reward)
    self.rewards[current_batch] += reward
    if 'default_TS_settings' in self.agent_settings:
      found_bo_params = self.agent_settings['default_TS_settings']['bo_params']

  def update_step(self, b_train, current_step):
    pass

  def update_round(self, b_train, info_dict, max_episode_length, current_round):
    """Update GP model, optimize acquisition function, and create new TS agents."""
    if current_round >= self.first_round_BO:
      # round=0 is for MRT, round=1 is for Sobol

      # construct Y
      new_obj = self.rewards.unsqueeze(-1).to(dtype=self.dtype, device=self.device)
      if self.Y is None:
        self.Y = new_obj
      else:
        self.Y = torch.cat((self.Y, new_obj), axis=0)
        if self.agent_name.find('TuRBO') >= 0:
          self.turbo_state = update_TurboState(self.turbo_state, new_obj)

      max_rounds_index = info_dict['ROUNDS']-1
      if current_round < max_rounds_index:

        chosen_new_batch_size = self.B_schedule[current_round+1]

        # fit GP
        train_X_GP = normalize(self.X, self.agent_settings['x_bounds'])
        train_y_GP = (self.Y - self.Y.mean()) / max(1e-5, self.Y.std())

        if self.agent_name.find('TuRBO') >= 0:
          # same as in TuRBO paper
          likelihood = GaussianLikelihood(noise_constraint=Interval(1e-8, 1e-3))          
          dim = train_X_GP.shape[-1]
          covar_module = ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=dim, lengthscale_constraint=Interval(0.005, 4.0)))
          self.model = model = SingleTaskGP(train_X_GP, train_y_GP, covar_module=covar_module, likelihood=likelihood)       
        else:
          likelihood = GaussianLikelihood()
          self.model = SingleTaskGP(train_X_GP, train_y_GP, likelihood=likelihood)

        mll = ExactMarginalLogLikelihood(self.model.likelihood, self.model)
        max_cholesky_size = float("inf")  # always use Cholesky
        with gpytorch.settings.max_cholesky_size(max_cholesky_size):
          fit_gpytorch_mll(mll)

        # optimize acq fn      
        if self.agent_name == 'BO-qEI':
          best_f = train_y_GP.max()
          acq_fn = qExpectedImprovement(model=self.model, best_f=best_f)            
          with gpytorch.settings.max_cholesky_size(max_cholesky_size):
            candidates, _ = optimize_acqf(acq_function=acq_fn, bounds=self.agent_settings['x_bounds_NORMALIZED'],
                                          q=chosen_new_batch_size, num_restarts=self.agent_settings['num_restarts'], raw_samples=self.agent_settings['raw_samples'])    

        elif self.agent_name.find('TuRBO') >= 0:
          if current_round <= self.first_round_BO+1:
            # round=0 is for MRT, round=1 is for Sobol
            best_f = train_y_GP.max()
            acq_fn = qExpectedImprovement(model=self.model, best_f=best_f)            
            with gpytorch.settings.max_cholesky_size(max_cholesky_size):
              candidates, _ = optimize_acqf(acq_function=acq_fn, bounds=self.agent_settings['x_bounds_NORMALIZED'],
                                            q=chosen_new_batch_size, num_restarts=self.agent_settings['num_restarts'], raw_samples=self.agent_settings['raw_samples'])
          else:
            if self.agent_name == 'TuRBO-qEI':
              CHOSEN_turbo_acq_fn_name = 'qEI'
            elif self.agent_name == 'TuRBO-TS':
              CHOSEN_turbo_acq_fn_name = 'TS'
            else:
              error_message = 'turbo_acq_fn_name = {} does not exist in BatchAgentBO update_round()!'.format(CHOSEN_turbo_acq_fn_name)
              assert(1==2), error_message
            candidates = generate_TuRBO_batch(  turbo_state=self.turbo_state,
                                                model=self.model,
                                                X=train_X_GP,
                                                Y=train_y_GP,
                                                batch_size=chosen_new_batch_size,
                                                seed_for_TS=int(self.agent_seed[0][0]),                                          
                                                num_restarts=self.agent_settings['num_restarts'], raw_samples=self.agent_settings['raw_samples'], max_cholesky_size=max_cholesky_size,
                                                dtype_for_TS=self.dtype, device_for_TS=self.device, 
                                                turbo_acq_fn_name=CHOSEN_turbo_acq_fn_name,                                            
                                                n_candidates_for_TS=self.n_candidates_for_TS
                                             )
        elif self.agent_name == 'Sobol':
          candidates = self.X_Sobol_norm[self.Sobol_count_so_far:self.Sobol_count_so_far+chosen_new_batch_size]
          self.Sobol_count_so_far += chosen_new_batch_size
        else:
          error_message = 'agent_name = {} does not exist in BatchAgentBO update_round()!'.format(self.agent_name)
          assert(1==2), error_message

        # construct X
        new_x_list = unnormalize(candidates.detach(), bounds=self.agent_settings['x_bounds'])
        self.X = torch.cat((self.X, new_x_list), axis=0)
        self.rewards = torch.zeros((chosen_new_batch_size), dtype=self.dtype, device=self.device)

        # create new agents using new X
        self.agents = []
        #same as for b in range(chosen_new_batch_size):
        for b_index, x_value in enumerate(new_x_list):
          input_dims = len(self.default_TS_settings['chosen_obs_names'])
          beta_coeffs, sigma_y, prior_coeff_mu_b, prior_coeff_mu_w, prior_coeff_sigma = get_TS_coeffs(self.default_TS_settings['bo_params'], 
                          x_value, input_dims, 
                          self.default_TS_settings['prior_coeff_mu_w'],  self.default_TS_settings['prior_coeff_mu_b'],  
                          self.default_TS_settings['prior_coeff_sigma'], self.default_TS_settings['sigma_y'], 
                          self.default_TS_settings['n_actions'])
          agent = ThompsonSamplingBayesLRAgent( n_arms=self.default_TS_settings['n_actions'], seed=self.agent_seed[current_round+1][b_index], 
                                                sigma_y=sigma_y, beta_coeffs=beta_coeffs,
                                                prior_coeff_mu_b=prior_coeff_mu_b, prior_coeff_mu_w=prior_coeff_mu_w, prior_coeff_sigma=prior_coeff_sigma,
                                                bo_params=self.default_TS_settings['bo_params'], max_beta=self.default_TS_settings['max_beta'])
          self.agents.append(agent)
      else:
        if current_round > max_rounds_index:
          error_message = 'ERROR! FOUND current_round={} and max_rounds_index={} in BatchAgentBO update_round()!'.format(current_round, max_rounds_index)
          print(error_message)
          assert(1==2), error_message
