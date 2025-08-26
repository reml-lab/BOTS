import numpy as np
import torch

def set_random_seed(seed):
    """Set the random seed for NumPy and PyTorch."""
    np.random.seed(seed)
    torch.manual_seed(seed)

def save_RL_npz(str_agent_type, reward_matrix, n_envs, env_version, env_sigma, output_folder):
    """Save reinforcement learning results to a .npz file."""
    outfile = output_folder + 'npz_v{}_sigma{}_N{}_{}.npz'.format(
        env_version, str(env_sigma).replace('0.','').replace('.',''), n_envs, str_agent_type
    )
    np.savez(outfile, results=reward_matrix)
  
def get_RL_results(str_agent_type, n_envs, env_version, env_sigma):
    """Load reinforcement learning results from a .npz file."""
    pathname = 'npz_v{}_sigma{}_N{}_{}.npz'.format(
        env_version, str(env_sigma).replace('0.','').replace('.',''), n_envs, str_agent_type
    )
    npzfile = np.load(pathname, allow_pickle=True)
    reward_matrix = npzfile['results']
    return reward_matrix

def triu_flatten(X):
    """Flatten the upper triangle of a batch of square matrices."""
    D = X.shape[-1]
    mask = torch.triu(torch.ones(D, D))==1
    return X[..., mask]

class Packing():
    """Class to handle indexing and extraction of parameters from BO input vector."""
    
    def __init__(self, bo_params, input_x, n_arms=4):
        """Initialize indices and store input vector."""
        self.n_arms = n_arms
        self.indices = {}; index_so_far = -1
        if bo_params.find('betas3') >= 0:
            self.indices['beta_coeffs'] = [index_so_far+1, index_so_far+2, index_so_far+3]
            index_so_far = self.indices['beta_coeffs'][-1]
        if bo_params.find('sigmaYs') >= 0:
            if bo_params.find('sigmaYs1') >= 0:
                self.indices['sigmaYs1'] = [index_so_far+1]
                index_so_far = self.indices['sigmaYs1'][-1]
            elif bo_params.find('sigmaYs4') >= 0:
                self.indices['sigmaYs4'] = [index_so_far+1, index_so_far+2, index_so_far+3, index_so_far+4]
                index_so_far = self.indices['sigmaYs4'][-1]
            else:
                assert(1==2), 'cannot find bo_params={} in Packing'.format(bo_params)
        self.CURRENT_MAX_INDEX = index_so_far
        assert(self.CURRENT_MAX_INDEX+1 == input_x.shape[-1]), \
            'error {} CURRENT_MAX_INDEX+1={} does not match input_x.shape[-1]={}'.format(
                bo_params, self.CURRENT_MAX_INDEX, input_x.shape[-1])
        self.create_x_full(input_x)

    def create_x_full(self, input_x):
        """Store the full input vector."""
        self.x_full = input_x
        return self.x_full

    def get_beta_coeffs(self):
        """Return the beta coefficients from input vector."""
        return self.x_full[..., self.indices['beta_coeffs']]

    def get_sigmaYs1(self):
        """Return sigmaYs1 parameter from input vector."""
        return self.x_full[..., self.indices['sigmaYs1']]

    def get_sigmaYs4(self):
        """Return sigmaYs4 parameters from input vector."""
        return self.x_full[..., self.indices['sigmaYs4']]

    def get_prior_coeff_mu_b(self, chosen_arm):
        """Return prior coefficient mu_b for a given arm."""
        return self.x_full[..., self.indices['prior_coeff_mu_b{}'.format(chosen_arm)]]

    def get_prior_coeff_mu_w(self, chosen_arm):
        """Return prior coefficient mu_w for a given arm."""
        return self.x_full[..., self.indices['prior_coeff_mu_w{}'.format(chosen_arm)]]

    def get_indices_w_arm(self, chosen_arm):
        """Return indices for mu_b and mu_w for a given arm."""
        return list(self.indices['prior_coeff_mu_b{}'.format(chosen_arm)]) + \
               list(self.indices['prior_coeff_mu_w{}'.format(chosen_arm)])

    def get_indices_uppers10_row(self, chosen_row):
        """Return indices for upper-triangle row in 10x10 matrix."""
        return list(self.indices['U_row{}'.format(chosen_row)])

    def get_indices_uppers10(self):
        """Return all indices for upper-triangle entries of 10x10 matrices."""
        all_uppers10_indices = []
        for row in range(self.n_arms):
            all_uppers10_indices += self.get_indices_uppers10_row(row)
        return all_uppers10_indices

    def get_indices_uppers10_arm(self, chosen_arm):
        """Return upper-triangle indices for a specific arm."""
        shifted = chosen_arm * 10
        uppers10_indices_arm = []
        for row in range(self.n_arms):
            uppers10_indices_arm += [x + shifted for x in self.indices['U_row{}'.format(row)]]
        return uppers10_indices_arm

    def get_flatten_U(self, chosen_arm):
        """Return flattened upper-triangle entries for a given arm."""
        U_indices = self.get_indices_uppers10_arm(chosen_arm)
        flatten_U = self.x_full[..., U_indices]
        return flatten_U

    def get_U(self, chosen_arm):
        """Return unflattened upper-triangle matrix for a given arm."""
        flatten_triu = self.get_flatten_U(chosen_arm)
        unflatten_u = triu_unflatten(flatten_triu, self.n_arms)
        return unflatten_u

    def get_Sigma(self, chosen_arm):
        """Return covariance matrix for a given arm."""
        unflatten_u = self.get_U(chosen_arm)
        cov_matrix = cov_from_triu(unflatten_u)
        return cov_matrix

def get_x_bounds(str_BO_PARAMS, BETA_MIN, BETA_MAX, SIGMA_Y_MIN, SIGMA_Y_MAX,
                  MU_MIN, MU_MAX, U_DIAG_MIN, U_DIAG_MAX, U_TOP_MIN, U_TOP_MAX, device, dtype):
    """Return dimension and bounds tensor for BO input vector."""
    betas01_MIN = [BETA_MIN] * 2
    betas01_MAX = [BETA_MAX] * 2
    betas3_MIN = [BETA_MIN] * 3
    betas3_MAX = [BETA_MAX] * 3
    mus4_MIN   = [MU_MIN] * 4
    mus4_MAX   = [MU_MAX] * 4
    mus16_MIN  = [MU_MIN] * 16
    mus16_MAX  = [MU_MAX] * 16
    triu_lo = torch.Tensor([[U_DIAG_MIN, U_TOP_MIN,  U_TOP_MIN,  U_TOP_MIN],
                            [0.,         U_DIAG_MIN, U_TOP_MIN,  U_TOP_MIN],
                            [0.,         0.,         U_DIAG_MIN, U_TOP_MIN],
                            [0.,         0.,         0.,         U_DIAG_MIN]])
    triu_hi = torch.Tensor([[U_DIAG_MAX, U_TOP_MAX,  U_TOP_MAX,  U_TOP_MAX],
                            [0.,         U_DIAG_MAX, U_TOP_MAX,  U_TOP_MAX],
                            [0.,         0.,         U_DIAG_MAX, U_TOP_MAX],
                            [0.,         0.,         0.,         U_DIAG_MAX]])
    triu_lo_flatten = triu_flatten(triu_lo)
    triu_hi_flatten = triu_flatten(triu_hi)
    x_dim = 0; x_bounds = None

    def update_x_bounds(x_bounds1, x_bounds2):
        """Concatenate new bounds to existing bounds tensor."""
        if x_bounds1 is None:
            x_bounds1 = x_bounds2
        else:
            x_bounds1 = torch.cat([x_bounds1, x_bounds2], axis=-1)
        return x_bounds1

    if str_BO_PARAMS.find('betas3') >= 0:
        x_dim += 3
        x_bounds_= torch.tensor([betas3_MIN, betas3_MAX], device=device, dtype=dtype)
        x_bounds = update_x_bounds(x_bounds, x_bounds_)   
    if str_BO_PARAMS.find('sigmaYs1') >= 0:
        x_dim += 1
        x_bounds_= torch.tensor([[SIGMA_Y_MIN], [SIGMA_Y_MAX]], device=device, dtype=dtype)
        x_bounds = update_x_bounds(x_bounds, x_bounds_)
    if str_BO_PARAMS.find('sigmaYs4') >= 0:
        x_dim += 4
        x_bounds_= torch.tensor([[SIGMA_Y_MIN] * 4, [SIGMA_Y_MAX] * 4], device=device, dtype=dtype)
        x_bounds = update_x_bounds(x_bounds, x_bounds_)
    if x_bounds is None:
        str_error_message = 'cannot find BO_PARAMS = {} in get_x_bounds()!'.format(str_BO_PARAMS)
        print('\n{}\n'.format(str_error_message))
        assert(1==2), str_error_message
    return x_dim, x_bounds

def get_TS_coeffs(bo_params, input_x, input_dims, PRIOR_COEFF_MU_w, PRIOR_COEFF_MU_b, PRIOR_COEFF_SIGMA, SIGMA_Y, n_arms):
    """Return beta coefficients, sigma_Y, and prior coefficients for TS."""
    pack = Packing(bo_params, input_x.unsqueeze(0))
    prior_coeff_mu_b = {}; prior_coeff_mu_w = {}; prior_coeff_sigma = {}
    if bo_params == 'betas3':
        beta_coeffs = pack.get_beta_coeffs().reshape(3)
        sigma_Y = SIGMA_Y
        for arm in range(n_arms):
            prior_coeff_mu_b[arm]  = PRIOR_COEFF_MU_b
            prior_coeff_mu_w[arm]  = PRIOR_COEFF_MU_w
            prior_coeff_sigma[arm] = PRIOR_COEFF_SIGMA
    elif (bo_params == 'betas3 sigmaYs1') or (bo_params == 'betas3 sigmaYs4'):
        beta_coeffs = pack.get_beta_coeffs().reshape(3)
        sigma_Y = float(pack.get_sigmaYs1()[0][0]) if (bo_params.find('sigmaYs1') >= 0) else pack.get_sigmaYs4().reshape(4)
        for arm in range(n_arms):
            prior_coeff_mu_b[arm]  = PRIOR_COEFF_MU_b
            prior_coeff_mu_w[arm]  = PRIOR_COEFF_MU_w
            prior_coeff_sigma[arm] = PRIOR_COEFF_SIGMA
    else:
        str_error_message = 'cannot find bo_params = {} in get_TS_coeffs!'.format(bo_params)
        print('\n{}\n'.format(str_error_message))
        assert(1==2), str_error_message
    return (beta_coeffs, sigma_Y, prior_coeff_mu_b, prior_coeff_mu_w, prior_coeff_sigma)
