import numpy as np
import matplotlib.pyplot as plt
import os

from BOTS.envs import StepCountJITAI


def generate_seeds(trials, rounds, batch_size, n_init, init_seed, max_int=2**31-1):
  """Generate random seeds for agents and environments."""
  np.random.seed(seed=init_seed); seed_matrix_dim = max(n_init, batch_size)
  seeds_agent = np.random.randint(low=0, high=max_int, size=(trials, rounds, seed_matrix_dim))
  seeds_env   = np.random.randint(low=0, high=max_int, size=(trials, rounds, seed_matrix_dim))
  return seeds_agent, seeds_env


def create_B_schedule(chosen_R, chosen_B, chosen_n_MRT, chosen_n_init):
  """Create batch size schedule for each round."""
  B_schedule = [None] * (chosen_R)
  B_schedule[0] = chosen_n_MRT
  B_schedule[1] = chosen_n_init
  for r in range(2, chosen_R):
    B_schedule[r] = chosen_B
  return B_schedule


def create_env_list(seeds_env, trial, chosen_round, batch_size, chosen_n_version, chosen_obs_names,
                    b_using_uniform=True, CHOSEN_a = .2,
                    CHOSEN_δh=0.1, CHOSEN_εh=0.05, CHOSEN_δd=0.1, CHOSEN_εd=0.1,  # default
                    chosen_sigma=.4, D_threshold=0.99, b_display=False):
  """Initialize a list of StepCountJITAI environment instances."""
  new_env_list = []
  for batch_index in range(batch_size):
    new_env = StepCountJITAI( sigma=chosen_sigma, chosen_obs_names=chosen_obs_names, n_version=chosen_n_version,
                              seed=seeds_env[trial][chosen_round][batch_index],
                              b_using_uniform=b_using_uniform, chosen_a=CHOSEN_a, 
                              δh=CHOSEN_δh, εh=CHOSEN_εh, δd=CHOSEN_δd, εd=CHOSEN_εd,
                              D_threshold=D_threshold, b_display=b_display)
    new_env_list.append(new_env)
  return new_env_list


def plot_batch_results(str_agent_name, y_train_values, output_detail, output_folder, chosen_BO_PARAMS, chosen_TRIALS, chosen_ROUNDS, chosen_BATCH_SIZE, chosen_STEPS, 
                       chosen_N_INIT, chosen_n_version, chosen_obs_names, b_update_stdout, compute_method='divide_by_batch', chosen_B_schedule=None,
                       b_mean_sd=True, color=None, str_file_type='.pdf', b_display=False, fig_size=(3.2,2.8), y_lim=(-100,3400), b_show_legend=False,
                       color_dict = {'TS':'c', 'TS_betas3':'c', 'TS_prior':'c', 'AlternateActions':'dimgrey', 'QLearning':'dodgerblue',
                                     'DQN':'g', 'REINFORCE':'b', 'PPO':'purple', 'BO-qEI':'m', 'TuRBO-qEI':'orangered', 'TuRBO-TS':'orangered', 'Sobol':'k'},
                       plot_title_detail='', x_label=None, y_label='Average Return'):
  """Plot batch results with optional mean and standard deviation shading."""
  # Note: compute_method should be set to divide_by_batch, and chosen_B_schedule must be set.
  str_key = '{} T{} R{} B{}'.format(str_agent_name, chosen_TRIALS, chosen_ROUNDS, chosen_BATCH_SIZE)
  if compute_method == 'steps':
    all_returns = y_train_values.sum(-1).mean(-1)
  elif compute_method == 'round':
    all_returns = y_train_values
  elif compute_method == 'divide_by_batch':
    if chosen_B_schedule is None:
      error_message = 'Attention! found compute_method = {} in plot_batch_results! chosen_B_schedule must be set!'.format(compute_method)
      print(error_message)
      assert(1==2), error_message
    else:
      chosen_B_schedule = np.array(chosen_B_schedule)
      all_returns = y_train_values / chosen_B_schedule
  else:
    error_message = 'compute_method = {}  does not exist in plot_batch_results! compute_method can be: steps or round!'.format(compute_method)
    print(error_message)
    assert(1==2), error_message
  if color is None:
    color = 'dimgrey'
    if str_agent_name in color_dict:
      color = color_dict[str_agent_name]
  plt.figure(figsize=fig_size)
  if b_mean_sd:
    all_returns_mean = all_returns.mean(axis=0)
    all_returns_sd   = all_returns.std(axis=0)
    xs = np.arange(all_returns_mean.shape[0])
    plt.plot(all_returns_mean, color=color, label='mean')
    plt.fill_between(xs, all_returns_mean-all_returns_sd, all_returns_mean+all_returns_sd, color=color, alpha=0.3)
  else:
    for trial in range(chosen_TRIALS):
      plt.plot(all_returns[trial], label='t{}'.format(trial))
      if b_display: print('{} returns[{}]\tmean={:.1f} \tsd={:.1f} \tshape={}'.format(str_agent_name, trial, all_returns[trial].mean(), all_returns[trial].std(), tuple(all_returns[trial].shape)))
  if len(plot_title_detail) < 1:
    str_plot_title_detail = '\n'
  else:
    str_plot_title_detail = '\n' + plot_title_detail + '\n'
  output_detail = output_detail.replace('sigma_s', 'σs')
  plot_title = '{}{}{}'.format(str_agent_name, str_plot_title_detail, output_detail.replace('_', ' '))
  plt.xlabel(x_label)
  plt.ylabel(y_label)
  plt.title(plot_title)
  plt.ylim(y_lim); plt.grid(); plt.tight_layout()
  if b_show_legend: plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
  plt.show()


