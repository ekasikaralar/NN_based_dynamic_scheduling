import numpy as np
import torch


class Equation(object):
    """Base class for defining PDE related function"""

    def __init__(self, eqn_config):

        self.dim = eqn_config.dim
        self.total_time = eqn_config.total_time

        self.num_time_interval = eqn_config.num_time_interval
        self.delta_t = self.total_time / self.num_time_interval
        self.sqrt_delta_t = np.sqrt(self.delta_t)

    
    def f_tf(self, x, a_lowbound, mu, theta, cost,  z, u):
        """Generator function in the PDE"""
        raise NotImplementedError
    
    def g_tf(self, x):
        """Terminal condition of the PDE"""
        raise NotImplementedError

class HJB(Equation):

    def __init__(self, eqn_config):
        
        super(HJB, self).__init__(eqn_config)
        
        self.policy = eqn_config.policy
        self.overtime_cost = eqn_config.overtime_cost
    
    def generate_u_sample(self, num_sample):
        
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        if self.policy == "even":

            u_sample = torch.ones(num_sample, self.dim).to(device) * 1 / self.dim 

        elif self.policy == "minimal":

            u_sample = torch.zeros(num_sample, self.dim).to(device) 

        elif self.policy == "best":

            u_sample = torch.zeros(num_sample, self.dim).to(device) 
            u_sample[:,1] = 1

        elif self.policy == "best_var":

            u_sample = torch.zeros(num_sample, self.dim).to(device)
            
            u_sample[:,0] = 1

        elif self.policy == "weighted_split_3class":

            u_sample = torch.ones(num_sample, self.dim).to(device) * 0.037178922
            u_sample[:,0:3] = 0.159803922

        elif self.policy == "weighted_split_4class":
            
            u_sample = torch.ones(num_sample, self.dim).to(device) * 0.0377
            u_sample[:, 0:3] = 0.1272
            u_sample[:,14] = 0.1272
        
    
        return u_sample

    def compute_mx(self, x, a_lowbound):
        
        sum_x = torch.sum(x, dim = 1, keepdim = True)
        sum_x_p = torch.clamp(sum_x, min = 0.0)
        sum_x_n = torch.clamp(sum_x, max = 0.0)

        return sum_x_p + a_lowbound * (torch.exp(sum_x_n) - 1)

        
    def f_tf(self, x, a_lowbound, mu, theta, cost,  z, u):
        
        mx = self.compute_mx(x, a_lowbound)
        
        first_term = torch.sum((mu - theta) * z * u, dim = 1, keepdim = True)
        
        second_term = torch.min(cost + (mu - theta) * z, dim = 1, keepdim = True)[0]
        
        return mx * (first_term - second_term)


    def g_tf(self, x):

        terminal = self.overtime_cost * torch.max(torch.sum(x, dim=1, keepdim = True), torch.tensor(0.0))

        return terminal