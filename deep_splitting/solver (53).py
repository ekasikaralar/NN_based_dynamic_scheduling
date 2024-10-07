import numpy as np
import torch
import time
from torch import nn
import os
import json
import pandas as pd


from Net import Y_Net
from Net import Z_Net

torch.manual_seed(73)

class DSPL_solver(object):

    def __init__(self, config, bsde):

        self.config = config
        
        self.net_config = config.net_config
        self.eqn_config = config.eqn_config

        self.bsde = bsde
        
        self.dim = self.eqn_config.dim
        self.num_time_interval = self.net_config.num_time_interval
        self.print_interval = self.net_config.print_interval

        self.num_neurons = self.net_config.num_neurons
        self.activation = self.net_config.activation
        self.num_layers = self.net_config.num_layers
        self.slope = self.net_config.slope
        
        self.gamma = self.net_config.gamma

        self.final_learning_rate = self.net_config.final_learning_rate
        self.inner_learning_rate = self.net_config.inner_learning_rate

        self.final_milestones = self.net_config.final_milestones
        self.inner_milestones = self.net_config.inner_milestones

        self.grad_clip = self.net_config.grad_clip
        self.grad_norm = self.net_config.grad_norm
        
        self.patience = self.net_config.patience
        self.min_delta = self.net_config.min_delta
        self.early_stopping = self.net_config.early_stopping

        self.covar_multiplier = self.net_config.covar_multiplier
        self.lambd_const = self.net_config.lambd_const
        self.a_lowbound = self.net_config.a_lowbound

        self.T = self.eqn_config.total_time
        
        self.delta_t = self.T / self.num_time_interval
        self.sqrt_delta_t = np.sqrt(self.delta_t)
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.mu = torch.tensor(pd.read_csv(self.eqn_config.mu_file, header = None).values, dtype = torch.float32).T.to(self.device)
        self.theta = torch.tensor(pd.read_csv(self.eqn_config.theta_file, header = None).values, dtype = torch.float32).T.to(self.device)
        self.cost = torch.tensor(pd.read_csv(self.eqn_config.cost_file, header = None).values, dtype = torch.float32).T.to(self.device)
        self.lambd = torch.tensor(pd.read_csv(self.eqn_config.lambd_file, header = None).values, dtype = torch.float32).T.to(self.device)
        self.zeta = torch.tensor(pd.read_csv(self.eqn_config.zeta_file, header = None).values, dtype = torch.float32).T.to(self.device)
        self.mean = torch.tensor(pd.read_csv(self.eqn_config.means_file, header = None).values, dtype = torch.float32).to(self.device)
        self.std = torch.tensor(pd.read_csv(self.eqn_config.stds_file, header = None).values, dtype = torch.float32).to(self.device)

        self.lambd = self.lambd.repeat(self.num_time_interval // 204, 1)
        self.zeta = self.zeta.repeat(self.num_time_interval // 204, 1)
        self.sigma = self.covar_multiplier * torch.sqrt(2 * self.lambd)
        
    
    # Helper function
    def calculate_negative_loss(self, func):

        zero_func = torch.min(func.min(dim = 1, keepdim = True)[0], torch.tensor(0.0))
        negative_loss = torch.sum(zero_func ** 2)
        
        return negative_loss

    # Generate sample paths        
    def sample(self, num_sample, n):
        
        u = self.bsde.generate_u_sample(num_sample) ## reference policy is independent of the interval

        x0 = (torch.rand(num_sample, self.dim).to(self.device) * 20) - 10
        
        if n != 0:

            for k in range(self.dim):

                x0[:, k] = torch.normal(mean=self.mean[k, n], std=self.std[k, n], size = (num_sample,))
                
        W = torch.randn(num_sample, self.dim).to(self.device) * self.sqrt_delta_t

        sum_x = torch.sum(x0, dim = 1, keepdim = True)
        mx = torch.clamp(sum_x, min = 0.0)

        drift = (self.zeta[n, :] - self.mu * x0) * self.delta_t + (mx * (self.mu - self.theta) * u) * self.delta_t
        diffu = self.sigma[n, :] * W

        x1 = x0 + drift + diffu
            
        return x0, x1, u
        
        
    def loss_fn(self, v_prev, v_next, negative_loss):

        delta = v_prev - v_next
        
        return torch.mean(delta ** 2) + self.lambd_const * negative_loss
    
    def save_weights_as_txt(self, network, network_name, save_path_cppweights):

        for j, layer in enumerate(network.linearlayers):

            # check if the layer has weights 
            if hasattr(layer, 'weight'):
                weight_filename = os.path.join(save_path_cppweights, f"{network_name}_w{j}.txt")
                np.savetxt(weight_filename, layer.weight.detach().cpu().numpy(), delimiter = ",")
            
            # check if the layer has bias
            if hasattr(layer, 'bias') and layer.bias is not None:
                bias_filename = os.path.join(save_path_cppweights, f"{network_name}_b{j}.txt")
                np.savetxt(bias_filename, layer.bias.detach().cpu().numpy(), delimiter = ",")
    
    def train(self, final_iterations, inner_iterations, save_path_logs, save_path_cppweights):

        y_net_list = nn.ModuleList([Y_Net(self.dim, self.num_neurons, self.num_layers, self.activation, self.slope) for _ in range(self.num_time_interval)])
        z_net_list = nn.ModuleList([Z_Net(self.dim, self.num_neurons, self.num_layers, self.activation, self.slope) for _ in range(self.num_time_interval)])

        y_net_list = y_net_list.to(self.device)
        z_net_list = z_net_list.to(self.device)

        start_time = time.time()
        self.valid_size = self.net_config.valid_size
        self.batch_size = self.net_config.batch_size

        # Training the networks at the (N-1)th interval
        optimizer = torch.optim.Adam(
                                    list(y_net_list[self.num_time_interval - 1].parameters()) + 
                                    list(z_net_list[self.num_time_interval - 1].parameters()), lr = self.final_learning_rate) 

        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones = self.final_milestones, gamma = self.gamma)
        
        training_history = []

        # Generate validation data for the (N-1)th interval X(t_{N-1}) and X(t_{N})
        xn_valid, xn1_valid, u_valid = self.sample(self.valid_size, self.num_time_interval - 1)

        for i in range(final_iterations):

            ## training 
            y_net_list[self.num_time_interval - 1].train()
            z_net_list[self.num_time_interval - 1].train()

            a_lowbound = torch.max(torch.tensor(1 - i / 1000), torch.tensor(0.0))

            ## generate training data for the (N-1)th interval X(t_{N-1}) and X(t_{N})
            xn_train, xn1_train, u_train = self.sample(self.batch_size, self.num_time_interval - 1)

            vn1_train = self.bsde.g_tf(xn1_train)

            zn_train = z_net_list[self.num_time_interval - 1](xn_train)
            vn_train = y_net_list[self.num_time_interval - 1](xn_train)

            rhs_train = vn_train + self.bsde.f_tf(xn_train, a_lowbound, self.mu, self.theta, self.cost, zn_train, u_train) * self.delta_t
            
            negative_loss_train = self.calculate_negative_loss(vn_train) + self.calculate_negative_loss(zn_train)
                        
            loss = self.loss_fn(vn1_train, rhs_train, negative_loss_train)
            
            optimizer.zero_grad()
            loss.backward()
            
            if self.grad_clip:
                
                    torch.nn.utils.clip_grad_norm_(list(y_net_list[self.num_time_interval - 1].parameters()) + 
                                list(z_net_list[self.num_time_interval - 1].parameters()), self.grad_norm)

            optimizer.step()
            loss_train = loss.item()
            scheduler.step()
            
            ## validation
            y_net_list[self.num_time_interval - 1].eval()
            z_net_list[self.num_time_interval - 1].eval()
            
            vn1_valid = self.bsde.g_tf(xn1_valid)

            zn_valid = z_net_list[self.num_time_interval - 1](xn_valid)
            vn_valid = y_net_list[self.num_time_interval - 1](xn_valid)

            rhs_valid = vn_valid + self.bsde.f_tf(xn_valid, a_lowbound, self.mu, self.theta, self.cost, zn_valid, u_valid) * self.delta_t

            negative_loss_valid = self.calculate_negative_loss(vn_valid) + self.calculate_negative_loss(zn_valid)
            
            loss_valid = self.loss_fn(vn1_valid, rhs_valid, negative_loss_valid)
            loss_valid = loss_valid.item()
            
            if i % self.print_interval == 0:

                elapsed_time = time.time() - start_time
                training_history.append([self.num_time_interval - 1, i, loss_train, loss_valid, elapsed_time])
                print("interval: ", self.num_time_interval - 1, " iter: ", i, " train_loss: ", loss_train, " validation_loss: ", loss_valid, " elapsed_time: ", elapsed_time)
        
        # Save the Nth network
        torch.save(y_net_list[self.num_time_interval - 1].state_dict(), save_path_logs + f"y_network{self.num_time_interval - 1}.pth")
        torch.save(z_net_list[self.num_time_interval - 1].state_dict(), save_path_logs + f"z_network{self.num_time_interval - 1}.pth")

        # Backward for all time intervals from N - 2 to 0
        for n in range(self.num_time_interval - 2, -1, -1):

            y_net_list[n].load_state_dict(y_net_list[n+1].state_dict()) # load the previous model for initialized parameters
            z_net_list[n].load_state_dict(z_net_list[n+1].state_dict()) # load the previous model for initialized parameters

            for param in y_net_list[n+1].parameters():
                param.requires_grad = False

            for param in z_net_list[n+1].parameters():
                param.requires_grad = False

            # Define optimizer for the t-th network
            optimizer = torch.optim.Adam(
                                        list(y_net_list[n].parameters()) +
                                        list(z_net_list[n].parameters()), lr = self.inner_learning_rate) 
            
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones = self.inner_milestones, gamma = self.gamma)
            
            # Generate validation data
            xn_valid, xn1_valid, u_valid = self.sample(self.valid_size, n)

            best_val_loss = float('inf')
            patience_counter = 0
            patience = self.patience  # Number of iterations with no improvement, after which training will be stopped
            min_delta = self.min_delta

            for i in range(inner_iterations):

                ## training 
                y_net_list[n].train()
                z_net_list[n].train()

                if self.a_lowbound:  
                    
                    a_lowbound = torch.max(torch.tensor(1 - i / 1000), torch.tensor(0.0))

                else: 
                    a_lowbound = 0

                ## generate training data
                xn_train, xn1_train, u_train = self.sample(self.batch_size, n)

                ####
                vn1_train = y_net_list[n](xn1_train)

                zn_train = z_net_list[n](xn_train)
                vn_train = y_net_list[n](xn_train)

                rhs_train = vn_train + self.bsde.f_tf(xn_train, a_lowbound, self.mu, self.theta, self.cost, zn_train, u_train) * self.delta_t

                negative_loss_train = self.calculate_negative_loss(vn_train) + self.calculate_negative_loss(zn_train)

                loss = self.loss_fn(vn1_train, rhs_train, negative_loss_train)

                optimizer.zero_grad()
                loss.backward()

                if self.grad_clip:
                    
                    torch.nn.utils.clip_grad_norm_(list(y_net_list[n].parameters()) + 
                                list(z_net_list[n].parameters()), self.grad_norm)
                
                optimizer.step()
                loss_train = loss.item()
                scheduler.step()

                # Validation step
                with torch.no_grad():
                    
                    y_net_list[n].eval()
                    z_net_list[n].eval()

                    vn1_valid = y_net_list[n](xn1_valid)

                    zn_valid = z_net_list[n](xn_valid)
                    vn_valid = y_net_list[n](xn_valid)

                    rhs_valid = vn_valid + self.bsde.f_tf(xn_valid, a_lowbound, self.mu, self.theta, self.cost, zn_valid, u_valid) * self.delta_t
                    negative_loss_valid = self.calculate_negative_loss(vn_valid) + self.calculate_negative_loss(zn_valid)

                    val_loss = self.loss_fn(vn1_valid, rhs_valid, negative_loss_valid)
                    loss_valid = val_loss.item()
                    
                # Apply early stopping only if enabled
                if self.early_stopping:
                    
                    if loss_valid < best_val_loss - min_delta:
                        best_val_loss = loss_valid
                        patience_counter = 0
                    
                    else:
                        patience_counter += 1

                    # Early stopping condition
                    if patience_counter > patience:
                        print(f"Early stopping at iteration {i}, time interval {n}")
                        break

                if i % self.print_interval == 0:

                    elapsed_time = time.time() - start_time
                    training_history.append([n, i, loss_train, loss_valid, elapsed_time])
                    print("interval: ", n, " iter: ", i, " train_loss: ", loss_train, " validation_loss: ", loss_valid, " elapsed_time: ", elapsed_time)

            
            torch.save(y_net_list[n].state_dict(), save_path_logs + f"y_network{n}.pth")
            torch.save(z_net_list[n].state_dict(), save_path_logs + f"z_network{n}.pth")

        # save neural network weights as .txt files for the C++ simulation
        for n in range(self.num_time_interval):
            self.save_weights_as_txt(z_net_list[n], f"z_network{n}", save_path_cppweights)
        
        return training_history
