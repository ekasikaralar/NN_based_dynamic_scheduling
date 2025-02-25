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
    """
    DSPL Solver Class for solving PDE-related systems using neural networks.

    Attributes:
        config (object): General configuration object containing network and equation parameters.
        bsde (object): Object implementing the PDE method (e.g., HJB, FBSDE).
    """
        
    def __init__(self, config, bsde):

        # Configuration objects
        self.config = config
        self.net_config = config.net_config  # Neural Network Hyperparameters
        self.eqn_config = config.eqn_config  # PDE-related system parameters
        self.bsde = bsde  # PDE method instance

        # Device setup
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # Use GPU if available, otherwise CPU

        # PDE dimensions and time parameters
        self.dim = self.eqn_config.dim  # Dimensionality of the PDE
        self.num_time_interval = self.net_config.num_time_interval  # Number of time intervals partitioning the time horizon
        self.delta_t = self.eqn_config.total_time / self.num_time_interval  # Length of each time partition
        self.sqrt_delta_t = np.sqrt(self.delta_t)  # Square root of time partition length
        self.T = self.eqn_config.total_time  # Total time horizon
        self.print_interval = self.net_config.print_interval

        # Neural Network architecture parameters
        self.num_neurons = self.net_config.num_neurons  # Neurons in each hidden layer
        self.num_layers = self.net_config.num_layers  # Number of hidden layers
        self.activation = self.net_config.activation  # Activation function for hidden layers
        self.slope = self.net_config.slope  # Alpha for Leaky ReLU or CELU activations
        self.valid_size = self.net_config.valid_size
        self.batch_size = self.net_config.batch_size
        self.prev_init = self.net_config.prev_init

        # Learning rate parameters
        self.gamma = self.net_config.gamma  # Decay rate for learning rate
        self.inner_learning_rate = self.net_config.inner_learning_rate  # Initial learning rate for inner intervals
        self.final_learning_rate = self.net_config.final_learning_rate  # Initial learning rate for the final interval
        self.inner_milestones = self.net_config.inner_milestones  # Milestones for adjusting learning rate (inner intervals)
        self.final_milestones = self.net_config.final_milestones  # Milestones for adjusting learning rate (final interval)

        # Gradient clipping parameters
        self.grad_clip = self.net_config.grad_clip  # Enable/disable gradient clipping
        self.grad_norm = self.net_config.grad_norm  # Maximum gradient norm for clipping

        # Early stopping parameters
        self.early_stopping = self.net_config.early_stopping  # Enable/disable early stopping
        self.patience = self.net_config.patience  # Iteration patience for early stopping
        self.min_delta = self.net_config.min_delta  # Minimum improvement for early stopping

        # PDE-specific parameters
        self.lambd_const = self.net_config.lambd_const  # Negative gradient approximation constant
        self.a_lowbound = self.net_config.a_lowbound  # Lower bound for compute_mx function

        # Load system data (rates, costs, and distributions)
        self.mu = torch.tensor(pd.read_csv(self.eqn_config.mu_file, header=None).values, dtype=torch.float32).T.to(self.device)  # Hourly service rates
        self.theta = torch.tensor(pd.read_csv(self.eqn_config.theta_file, header=None).values, dtype=torch.float32).T.to(self.device)  # Hourly abandonment rates
        self.cost = torch.tensor(pd.read_csv(self.eqn_config.cost_file, header=None).values, dtype=torch.float32).T.to(self.device)  # Hourly cost rates
        self.lambd = torch.tensor(pd.read_csv(self.eqn_config.lambd_file, header=None).values, dtype=torch.float32).T.to(self.device)  # Hourly arrival rates
        self.zeta = torch.tensor(pd.read_csv(self.eqn_config.zeta_file, header=None).values, dtype=torch.float32).T.to(self.device)  # Second-order drift terms
        self.mean = torch.tensor(pd.read_csv(self.eqn_config.means_file, header=None).values, dtype=torch.float32).to(self.device)  # Means of sample paths
        self.std = torch.tensor(pd.read_csv(self.eqn_config.stds_file, header=None).values, dtype=torch.float32).to(self.device)  # Standard deviations of sample paths
        
        # Process repeated rates for partitioned time intervals
        self.lambd = self.lambd.repeat(self.num_time_interval // 204, 1)
        self.zeta = self.zeta.repeat(self.num_time_interval // 204, 1)
        self.sigma = torch.sqrt(2 * self.lambd)  # Diffusion coefficient for the state process

    
    # Helper function
    def calculate_negative_loss(self, func):
        """
        Calculate the negative loss as the sum of squared negative values in the function.

        Args:
            func (torch.Tensor): Input tensor to calculate the negative loss.

        Returns:
            torch.Tensor: The calculated negative loss.
        """

        zero_func = torch.clamp(func.min(dim=1, keepdim=True)[0], max=0.0)  # Keep only negative values
        negative_loss = torch.sum(zero_func ** 2)  # Sum of squares of negative values
        return negative_loss
        
    # Generate sample paths        
    def sample(self, num_sample, n):
        """
        Generate sample paths for the PDE system.

        Args:
            num_sample (int): Number of samples to generate.
            n (int): Current time interval index.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Current state (x_n),
            next state (x_{n+1}), and policy (u) samples.
        """
        # Generate the reference policy (independent of the interval)
        u = self.bsde.generate_u_sample(num_sample) ## reference policy is independent of the interval
        
        # Initial state (x0): Uniform distribution between -10 and 10
        x0 = (torch.rand(num_sample, self.dim).to(self.device) * 20) - 10
        
        # Adjust current state based on mean and std if not the first interval
        if n != 0:
            for k in range(self.dim):
                x0[:, k] = torch.normal(mean=self.mean[k, n], std=self.std[k, n], size = (num_sample,))
        
        # Generate random noise (W) for diffusion
        W = torch.randn(num_sample, self.dim).to(self.device) * self.sqrt_delta_t

        # Compute drift and diffusion terms
        sum_x = torch.sum(x0, dim = 1, keepdim = True)
        mx = torch.clamp(sum_x, min = 0.0)

        drift = (self.zeta[n, :] - self.mu * x0) * self.delta_t + (mx * (self.mu - self.theta) * u) * self.delta_t
        diffu = self.sigma[n, :] * W

        # Compute the next state
        x1 = x0 + drift + diffu
            
        return x0, x1, u
        
        
    def loss_fn(self, v_prev, v_next, negative_loss):
        """
        Compute the loss function.

        Args:
            v_prev (torch.Tensor): Predicted value from the previous time step.
            v_next (torch.Tensor): Predicted value from the next time step.
            negative_loss (torch.Tensor): Negative loss term to regularize the solution.

        Returns:
            torch.Tensor: Computed loss value.
        """
        # Difference between consecutive predictions
        delta = v_prev - v_next

        # Mean squared error with a regularization term
        
        return torch.mean(delta ** 2) + self.lambd_const * negative_loss
    
    def save_weights_as_txt(self, network, network_name, save_path_cppweights):
        # Iterate through each layer in the network
        for j, layer in enumerate(network.linearlayers):
            # Save weights
            if hasattr(layer, 'weight'):
                weight_filename = os.path.join(save_path_cppweights, f"{network_name}_w{j}.txt")
                np.savetxt(weight_filename, layer.weight.detach().cpu().numpy(), delimiter = ",")
            # Save biases if they are not None
            if hasattr(layer, 'bias') and layer.bias is not None:
                bias_filename = os.path.join(save_path_cppweights, f"{network_name}_b{j}.txt")
                np.savetxt(bias_filename, layer.bias.detach().cpu().numpy(), delimiter = ",")
    
    def train(self, final_iterations, inner_iterations, save_path_logs, save_path_cppweights):
        
        # Step 1: Initialize neural networks for all intervals
        y_net_list = nn.ModuleList([Y_Net(self.dim, self.num_neurons, self.num_layers, self.activation, self.slope) for _ in range(self.num_time_interval)]).to(self.device) # Network for value function approximation for each interval
        z_net_list = nn.ModuleList([Z_Net(self.dim, self.num_neurons, self.num_layers, self.activation, self.slope) for _ in range(self.num_time_interval)]).to(self.device)  # Network for gradient approximation for each interval

        training_history = []
        start_time = time.time()
        self.valid_size = self.net_config.valid_size
        self.batch_size = self.net_config.batch_size

        # Initialize optimizer and learning rate scheduler
        optimizer = torch.optim.Adam(
                                    list(y_net_list[self.num_time_interval - 1].parameters()) + 
                                    list(z_net_list[self.num_time_interval - 1].parameters()), lr = self.final_learning_rate) 

        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones = self.final_milestones, gamma = self.gamma)
        
        # Generate validation data for the interval
        xn_valid, xn1_valid, u_valid = self.sample(self.valid_size, self.num_time_interval - 1)

        # Step 2: Train the network for the final interval
        for i in range(final_iterations):

            # Training step
            y_net_list[self.num_time_interval - 1].train()
            z_net_list[self.num_time_interval - 1].train()

            # Compute a lower bound for constraint enforcement
            a_lowbound = torch.max(torch.tensor(1 - i / 1000), torch.tensor(0.0))

            # Generate training data
            xn_train, xn1_train, u_train = self.sample(self.batch_size, self.num_time_interval - 1)

            # Compute target and predictions
            vn1_train = self.bsde.g_tf(xn1_train)
            zn_train = z_net_list[self.num_time_interval - 1](xn_train)
            vn_train = y_net_list[self.num_time_interval - 1](xn_train)
            
            # Compute the right-hand side of the PDE
            rhs_train = vn_train + self.bsde.f_tf(xn_train, a_lowbound, self.mu, self.theta, self.cost, zn_train, u_train) * self.delta_t
            
            # Compute the loss
            negative_loss_train = self.calculate_negative_loss(vn_train) + self.calculate_negative_loss(zn_train)          
            loss = self.loss_fn(vn1_train, rhs_train, negative_loss_train)
            loss_train = loss.item()

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            
            # Apply gradient clipping if enabled
            if self.grad_clip:
                    torch.nn.utils.clip_grad_norm_(list(y_net_list[self.num_time_interval - 1].parameters()) + 
                                list(z_net_list[self.num_time_interval - 1].parameters()), self.grad_norm)

            
            optimizer.step()
            scheduler.step()
            
            # Validation step
            with torch.no_grad():
                y_net_list[self.num_time_interval - 1].eval()
                z_net_list[self.num_time_interval - 1].eval()
            
                vn1_valid = self.bsde.g_tf(xn1_valid)
                zn_valid = z_net_list[self.num_time_interval - 1](xn_valid)
                vn_valid = y_net_list[self.num_time_interval - 1](xn_valid)

                rhs_valid = vn_valid + self.bsde.f_tf(xn_valid, a_lowbound, self.mu, self.theta, self.cost, zn_valid, u_valid) * self.delta_t

                negative_loss_valid = self.calculate_negative_loss(vn_valid) + self.calculate_negative_loss(zn_valid)
                loss_valid = self.loss_fn(vn1_valid, rhs_valid, negative_loss_valid)
                loss_valid = loss_valid.item()
            
            # Log progress
            if i % self.print_interval == 0:
                elapsed_time = time.time() - start_time
                training_history.append([self.num_time_interval - 1, i, loss_train, loss_valid, elapsed_time])
                print("interval: ", self.num_time_interval - 1, " iter: ", i, " train_loss: ", loss_train, " validation_loss: ", loss_valid, " elapsed_time: ", elapsed_time)
        
        # Save the trained model for the current interval
        torch.save(y_net_list[self.num_time_interval - 1].state_dict(), save_path_logs + f"y_network{self.num_time_interval - 1}.pth")
        torch.save(z_net_list[self.num_time_interval - 1].state_dict(), save_path_logs + f"z_network{self.num_time_interval - 1}.pth")

        # Step 3: Train networks for the remaining intervals in reverse order
        for n in range(self.num_time_interval - 2, -1, -1):
            
            # Load weights from the next interval and freeze them
            if self.prev_init:
                y_net_list[n].load_state_dict(y_net_list[n+1].state_dict()) 
                z_net_list[n].load_state_dict(z_net_list[n+1].state_dict()) 

            for param in y_net_list[n+1].parameters():
                param.requires_grad = False

            for param in z_net_list[n+1].parameters():
                param.requires_grad = False

            # Initialize optimizer and learning rate scheduler
            optimizer = torch.optim.Adam(
                                        list(y_net_list[n].parameters()) +
                                        list(z_net_list[n].parameters()), lr = self.inner_learning_rate) 
            
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones = self.inner_milestones, gamma = self.gamma)
            
            # Generate validation data for the interval
            xn_valid, xn1_valid, u_valid = self.sample(self.valid_size, n)

            best_val_loss = float('inf') # Track the best validation loss
            patience_counter = 0 # Counter for early stopping
            patience = self.patience  # Number of iterations with no improvement, after which training will be stopped
            min_delta = self.min_delta

            for i in range(inner_iterations):

                # Training step
                y_net_list[n].train()
                z_net_list[n].train()
                
                # Compute a lower bound for constraint enforcement
                a_lowbound = torch.max(torch.tensor(1 - i / 1000), torch.tensor(0.0)) if self.a_lowbound else 0

                # Generate training data
                xn_train, xn1_train, u_train = self.sample(self.batch_size, n)

                # Compute target and predictions
                vn1_train = y_net_list[n+1](xn1_train) 
                zn_train = z_net_list[n](xn_train) 
                vn_train = y_net_list[n](xn_train)

                # Compute the right-hand side of the PDE
                rhs_train = vn_train + self.bsde.f_tf(xn_train, a_lowbound, self.mu, self.theta, self.cost, zn_train, u_train) * self.delta_t
                
                # Compute the loss
                negative_loss_train = self.calculate_negative_loss(vn_train) + self.calculate_negative_loss(zn_train)
                loss = self.loss_fn(vn1_train, rhs_train, negative_loss_train)
                loss_train = loss.item()

                # Backpropagation
                optimizer.zero_grad()
                loss.backward()
                
                # Apply gradient clipping if enabled
                if self.grad_clip:
                    torch.nn.utils.clip_grad_norm_(list(y_net_list[n].parameters()) + 
                                list(z_net_list[n].parameters()), self.grad_norm)
                
                optimizer.step()
                scheduler.step()

                # Validation step
                with torch.no_grad():   
                    y_net_list[n].eval()
                    z_net_list[n].eval()

                    vn1_valid = y_net_list[n+1](xn1_valid) 
                    zn_valid = z_net_list[n](xn_valid) 
                    vn_valid = y_net_list[n](xn_valid)

                    rhs_valid = vn_valid + self.bsde.f_tf(xn_valid, a_lowbound, self.mu, self.theta, self.cost, zn_valid, u_valid) * self.delta_t
                    negative_loss_valid = self.calculate_negative_loss(vn_valid) + self.calculate_negative_loss(zn_valid)

                    val_loss = self.loss_fn(vn1_valid, rhs_valid, negative_loss_valid)
                    loss_valid = val_loss.item()
                    
                # Handle early stopping
                if self.early_stopping:
                    if loss_valid < best_val_loss - min_delta:
                        best_val_loss = loss_valid
                        patience_counter = 0  
                    else:
                        patience_counter += 1

                    if patience_counter > patience:
                        print(f"Early stopping at iteration {i}, time interval {n}")
                        break
                
                # Log progress
                if i % self.print_interval == 0:
                    elapsed_time = time.time() - start_time
                    training_history.append([n, i, loss_train, loss_valid, elapsed_time])
                    print("interval: ", n, " iter: ", i, " train_loss: ", loss_train, " validation_loss: ", loss_valid, " elapsed_time: ", elapsed_time)

            # Save the trained model for the current interval
            torch.save(y_net_list[n].state_dict(), save_path_logs + f"y_network{n}.pth")
            torch.save(z_net_list[n].state_dict(), save_path_logs + f"z_network{n}.pth")

        # Step 4: Save the final weights for use in C++ simulation
        for n in range(self.num_time_interval):
            self.save_weights_as_txt(z_net_list[n], f"z_network{n}", save_path_cppweights)
        
        return training_history
