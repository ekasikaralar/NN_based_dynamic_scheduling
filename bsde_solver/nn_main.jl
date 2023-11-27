RUN_NAME = "neural_network_policy_name"

using CUDA
using cuDNN
CUDA.allowscalar(false)
using Random, Flux, DataFrames, StatsBase, JLD2
using JSON, CSV, TickTock, DelimitedFiles, NPZ
using Distributions
using JLD2: @save
Random.seed!(73)
device = gpu

# reading the configuration file

config = JSON.parsefile("config.json")


# neural network parameters

neural_network_params = config["neural_network_parameters"]

LAMBDA = neural_network_params["LAMBDA"] # negative gradient penalty coefficient

MINS_IN_HOUR = 60 # number of minutes in an hour
NUMBER_HOUR = neural_network_params["HOURS"] # the length of the time horizon in hours
PRECISION = neural_network_params["PRECISION"] # the length of the time intervals (5 minute precision)

# TOTAL_TIME = NUMBER_HOUR * MINS_IN_HOUR/PRECISION
TOTAL_TIME = neural_network_params["TOTAL_TIME"] # the length of the time horizon in terms of precision

DIM = neural_network_params["DIM"] # the dimension of the problem

NUM_TIME_INTERVAL = neural_network_params["NUM_TIME_INTERVAL"] # number of neural networks to use (discretization scheme)
PRINT_INTERVAL = neural_network_params["PRINT_INTERVAL"] # scaling factor to print the loss values

NUM_ITERATIONS = neural_network_params["NUM_ITERATIONS"] # number of neural network training iterations
NUM_NEURONS = neural_network_params["NUM_NEURONS"] # number of neurons in each hidden layer
# NUM_HIDDEN_LAYERS = neural_network_params["NUM_HIDDEN_LAYERS"] # number of hidden layers
COVAR_MULTIPLIER = neural_network_params["COVAR_MULTIPLIER"] # scaling factor to scale the diffusion coefficient

BATCH_SIZE = neural_network_params["BATCH_SIZE"] # size of the training batch 
VALID_SIZE = neural_network_params["VALID_SIZE"] # size of the validation batch

# piecewise decay learning rates

LEARNING_RATES = neural_network_params["LEARNING_RATES"] # the learning rates
DECAY_RATES = neural_network_params["DECAY_RATES"] # the steps to decay the learning rates

# system parameters

system_params = config["system_parameters"]

SCALING_FACTOR = MINS_IN_HOUR // PRECISION
MU = Float32.(Matrix(CSV.read(system_params["MU_FILE"], DataFrame, header=0)) / SCALING_FACTOR) |> device # MU_FILE contains hourly service rates
THETA = Float32.(Matrix(CSV.read(system_params["THETA_FILE"], DataFrame, header=0)) / SCALING_FACTOR) |> device # THETA_FILE contains hourly abandonment rates
COST = Float32.(Matrix(CSV.read(system_params["COST_FILE"], DataFrame, header=0)) / SCALING_FACTOR) |> device # COST_FILE contains hourly cost rates
LAMBD = Float32.(Matrix(CSV.read(system_params["LAMBD_FILE"], DataFrame, header=0)) / SCALING_FACTOR) |> device # LAMBD_FILE contains hourly limiting arrival rates
ZETA = Float32.(Matrix(CSV.read(system_params["ZETA_FILE"], DataFrame, header=0)) / SCALING_FACTOR) |> device # ZETA_FILE contrains hourly second order term zetas
OVERTIME_COST = system_params["OVERTIME_COST"] # overtime cost rate to calculate the loss
POLICY = system_params["POLICY"] # reference policy

# repeating data based on the number of intervals 

LAMBD = repeat(LAMBD, inner=[1,NUM_TIME_INTERVAL÷TOTAL_TIME]) |> device
ZETA = repeat(ZETA, inner=[1,NUM_TIME_INTERVAL÷TOTAL_TIME]) |> device
SIGMA = COVAR_MULTIPLIER .* sqrt.(2 * LAMBD) # diffusion coefficient 


# discretization scheme

DELTA_T = TOTAL_TIME / NUM_TIME_INTERVAL
SQRT_DELTA_T = sqrt(DELTA_T)

# for random behavior policy

DIRICHLET = Dirichlet(ones(DIM))

# activation function

function leakyrelu_manual(x)
    leakyrelu(x, 0.2)
end

"""
`createDeepNNChain(dim, layers, units, activation, output_units, output_activation, bias)`
Creates a deep neural network chain with the specified configuration.

# Arguments
- `dim::Int`: Input dimension.
- `layers::Int`: Number of hidden layers.
- `units::Int`: Number of units in each hidden layer.
- `activation`: Activation function for the hidden layers.
- `output_units::Int`: Number of units in the output layer.
- `output_activation`: Activation function for the output layer.
- `bias::Bool`: Whether to include bias in the output layer.

# Returns
- `Chain`: A Flux Chain representing the neural network.
"""
function createDeepNNChain(dim, units, activation, output_units, output_activation)
    
    return Chain(BatchNorm(dim, ϵ=1e-6, momentum=0.99),
                 Dense(dim, units, activation),
                 Dense(units, units, activation),
                 Dense(units, units, activation),
                 Dense(units, units, activation),
                 Dense(units, output_units, output_activation, bias=false))
end


# DeepNN_z is the neural network to estimate the gradient of the value function at each interval
DeepNN_z = [createDeepNNChain(DIM, NUM_NEURONS, leakyrelu_manual, DIM, identity) for _ in 1:NUM_TIME_INTERVAL] .|> device

# DeepNN_y is the neural network to estimate the value function at t = 0
DeepNN_y = createDeepNNChain(DIM, NUM_NEURONS, leakyrelu_manual, 1, identity) |> device



struct NonsharedModel
    deepnn_z::Vector{Chain}
    deepnn_y::Chain
end

function (m::NonsharedModel)(a_lowbound, dw, x, u)
    
    negative_loss = 0.0
    y = m.deepnn_y(x[:,1,:])
    z = m.deepnn_z[1](x[:,1,:])
    
    negative_loss += calculate_negative_loss(y) 
    negative_loss += calculate_negative_loss(z) 

    for t in 1:NUM_TIME_INTERVAL-1

        mx = compute_mx(x[:,t,:], a_lowbound)
        first_term = sum((MU - THETA).*z.*u, dims=1)
        second_term = minimum(COST .+ (MU-THETA).*z, dims=1)
        w = mx .* (first_term - second_term)
        
        y = y .+ DELTA_T .* w .+ sum(SIGMA[:,t].* z .* dw[:, t, :], dims=1)
        z = m.deepnn_z[t+1](x[:,t+1,:])

        negative_loss += calculate_negative_loss(z) 
    end
    
    mx = compute_mx(x[:,end,:], a_lowbound)
    first_term = sum((MU - THETA).*z.*u, dims=1)
    second_term = minimum(COST .+ (MU-THETA).*z, dims=1)
    w = mx .* (first_term - second_term)

    y = y .+ DELTA_T * w .+ sum(SIGMA[:,end].* z .* dw[:, end, :], dims=1)
    
    return y, negative_loss
end


# Helper functions
function calculate_negative_loss(func)
    zero_func = min.(minimum(func, dims=1), 0.0)
    negative_loss = sum(zero_func.^2)
    return negative_loss
end

function compute_mx(x_slice, a_lowbound)
    sum_x = sum(x_slice, dims=1)
    sum_x_p = max.(sum_x, 0.0)
    sum_x_n = min.(sum_x, 0.0)
    return sum_x_p + a_lowbound .* (exp.(sum_x_n) .- 1)
end


Flux.@functor NonsharedModel

"""
    sample(num_sample::Int) -> Tuple{Array{Float32, 3}, Array{Float32, 3}, Array{Float32, 2}}

Generate samples for stochastic processes.

# Arguments
- `num_sample::Int`: Number of samples to generate.

# Returns
- `dw_sample`: Sample increments of the Wiener process.
- `x_sample`: Sample paths of the stochastic process. (training and validation data)
- `u_sample`: Sample control process.
""" 
function sample(num_sample)
    dw_sample = generate_dw_sample(num_sample)
    u_sample = generate_u_sample(num_sample)

    x_sample = zeros(Float32, DIM, NUM_TIME_INTERVAL + 1, num_sample) |> device
    x_sample[:,1,:] = rand(DIM, num_sample) * 20 .- 10 # initialize it as a uniform (-10, 10) r.v.
    
    for i in 1:NUM_TIME_INTERVAL
        sum_x = sum(x_sample[:,i,:], dims=1)
        mx = max.(sum_x, 0.0)
        x_sample[:,i+1,:] = x_sample[:,i,:] + (ZETA[:,i] .- MU.*x_sample[:,i,:])*DELTA_T + SIGMA[:,i].*dw_sample[:,i,:] + (mx .* ((MU - THETA) .* u_sample)) .* DELTA_T
    end

    return dw_sample, x_sample, u_sample
end

# Function to generate sample increments of the Wiener process.
function generate_dw_sample(num_sample::Int)
    return (randn(DIM, NUM_TIME_INTERVAL, num_sample) |> device) * SQRT_DELTA_T
end

# Function to generate sample control process.
function generate_u_sample(num_sample::Int)

    if POLICY == "even" #evenly split behavior policy
        u_sample = ones(Float32, DIM, num_sample) |> device
        u_sample[:, :] .= 1 / DIM
    elseif POLICY == "random" #random behavior policy
        u_sample = Float32.(rand(DIRICHLET, num_sample)) |> device 
    elseif POLICY == "minimal" #minimal behavior policy
        u_sample = zeros(Float32, DIM, num_sample) |> device
    elseif POLICY == "weighted_split" #weighted split behavior policy
        u_sample = ones(Float32, DIM, num_sample) * 0.46/14 |> device
        u_sample[1,:] .= 0.18
        u_sample[2,:] .= 0.18
        u_sample[3,:] .= 0.18
    elseif POLICY == "best" # best static for 3 dimensional problem
        u_sample = zeros(Float32, DIM, num_sample) |> device
        u_sample[2,:] .= 1
    elseif POLICY =="best_var" # best static for 3 dimensional variant problem
        u_sample = zeros(Float32, DIM, num_sample) |> device
        u_sample[1,:] .= 1
    end 

    return u_sample
end


"""
    loss_fn(model, a, dw, x, u; training=true) -> Float64

Calculate the loss function for the given model and data.

# Arguments
- `model`: The model to evaluate.
- `a`: Parameter 'a' used in the model.
- `dw`: The increments of the Wiener process.
- `x`: The state process.
- `u`: The control process.
- `training`: Flag indicating whether the model is in training mode.

# Returns
- The calculated loss value
"""
function loss_fn(model, a, dw, x, u, training=true)
    
    # Toggle model mode based on training flag
    training ? Flux.trainmode!(model) : Flux.testmode!(model)

    # Compute model output
    y_terminal, negative_loss = model(a, dw, x, u)

    # Calculate loss components
    g_tf = OVERTIME_COST * max.(sum(x[:, end, :], dims=1), 0.0)
    delta = y_terminal - g_tf

    # Final loss calculation
    return mean(delta .^ 2) + LAMBDA * negative_loss
end

# optimizer`: A Flux optimizer composed of two parts:
#  - `ClipNorm(15)`: This applies gradient clipping to limit the maximum norm of the gradient to 15. 
#  - `Adam(1e-2, (0.9, 0.999), 1.0e-7)`: The Adam optimization algorithm with a learning rate of 0.01, β1 = 0.9, β2 = 0.999, and ε = 1.0e-7. 
global_model = NonsharedModel(DeepNN_z, DeepNN_y)
optimizer = Flux.Optimiser(ClipNorm(15), Adam(LEARNING_RATES[1], (0.9, 0.999), 1.0e-7))
opt_state = Flux.setup(optimizer, global_model) # The state of the optimizer with respect to the `global_model`. It is initialized using `Flux.setup`, which prepares the optimizer to be used with the given model.

tick()

training_history = zeros(Float32, NUM_ITERATIONS + 1, 4)
dw_sample_valid, x_sample_valid, u_sample_valid = sample(VALID_SIZE)

for step in 0:NUM_ITERATIONS

    dw_sample, x_sample, u_sample = sample(BATCH_SIZE)
    a_lowbound = max(1 - step/3000, 0)

    # calculating the validation loss 
    valid_loss = loss_fn(global_model, a_lowbound, dw_sample_valid, x_sample_valid, u_sample_valid, false)
    elapsed_time = peektimer()
    
    # adjusting the learning rate
    for rate in 1:length(DECAY_RATES)
        if step == DECAY_RATES[rate]
            Flux.adjust!(opt_state, LEARNING_RATES[rate+1])
        end
    end
    # calculating the training loss
    train_loss, grads = Flux.withgradient(global_model) do m
        loss_fn(m, a_lowbound, dw_sample, x_sample, u_sample)
    end
    Flux.update!(opt_state, global_model, grads[1])
    
    if step % 1 == 0
        println("Step: $step, Elapsed: $elapsed_time, Loss: $train_loss, Valid Loss: $valid_loss")
    end
    
    # recording the training data          
    training_history[step+1,:] = [step, train_loss, valid_loss, elapsed_time]
end


# saving the neural network weights

if !isdir("logs_" * RUN_NAME)
    mkdir("logs_" * RUN_NAME)
end
writedlm("logs_" * RUN_NAME * "/training_history.csv", training_history, ',')

z_nets = cpu.(global_model.deepnn_z)
y_net = cpu(global_model.deepnn_y)

for i in 1:NUM_TIME_INTERVAL
    jldsave("logs_" * RUN_NAME * "/z$i.jld", model_state = Flux.state(z_nets[i]))
end

jldsave("logs_" * RUN_NAME * "/y.jld", model_state = Flux.state(y_net))

# saving the optimizer for retraining the neural network 

@save "logs_" * RUN_NAME * "/final_optimizer.jld" optimizer


if !isdir("weights_" * RUN_NAME)
    mkdir("weights_" * RUN_NAME)
end

Flux.testmode!(y_net)

tuple_len(::NTuple{N, Any}) where {N} = N

for i in 1:NUM_TIME_INTERVAL
    for (j, layer) in enumerate(z_nets[i].layers)
        params = Flux.params(layer)
        if j == 1
            A = transpose(hcat(layer.γ, layer.β, layer.μ, layer.σ²))
        elseif j < tuple_len(z_nets[i].layers)
            A = transpose(hcat(params[1], params[2]))
        else
            A = transpose(params[1])
        end
        npzwrite("weights_" * RUN_NAME * "/z" * "$i" * "_layer" * "$j" * ".npy", A)
    end
end


# saving the neural network weights
for (j, layer) in enumerate(y_net.layers)
    params = Flux.params(layer)
    if j == 1
        A = transpose(hcat(layer.γ, layer.β, layer.μ, layer.σ²))
    elseif j < tuple_len(y_net.layers)
        A = transpose(hcat(params[1], params[2]))
    else
        A = transpose(params[1])
    end
    npzwrite("weights_" * RUN_NAME * "/y_layer" * "$j" * ".npy", A)
end


if !isdir("cppweights_" * RUN_NAME)
    mkdir("cppweights_" * RUN_NAME)
end

# saving the neural network weights to use in the C++ code to run the discrete event simulation (integrates neural network approximation with the C++ code)
for i in 1:NUM_TIME_INTERVAL
    for (j, layer) in enumerate(z_nets[i].layers)
        params = Flux.params(layer)
        if j == 1
            writedlm("cppweights_" * RUN_NAME * "/z" * "$i" * "_gamma.txt", transpose(layer.γ), ' ')
            writedlm("cppweights_" * RUN_NAME * "/z" * "$i" * "_beta.txt", transpose(layer.β), ' ')
            writedlm("cppweights_" * RUN_NAME * "/z" * "$i" * "_mu.txt", transpose(layer.μ), ' ')
            writedlm("cppweights_" * RUN_NAME * "/z" * "$i" * "_batchnorm_denom.txt", transpose(sqrt.(layer.σ² .+ 1e-6)), ' ')
        elseif j < tuple_len(z_nets[i].layers)
            writedlm("cppweights_" * RUN_NAME * "/z" * "$i" * "_w" * "$(j-1)" * ".txt", transpose(params[1]), ' ')
            writedlm("cppweights_" * RUN_NAME * "/z" * "$i" * "_b" * "$(j-1)" * ".txt", transpose(params[2]), ' ')
        else
            writedlm("cppweights_" * RUN_NAME * "/z" * "$i" * "_w" * "$(j-1)" * ".txt", transpose(params[1]), ' ')
        end
    end
end

for (j, layer) in enumerate(y_net.layers)
    params = Flux.params(layer)
    if j == 1
        writedlm("cppweights_" * RUN_NAME * "/y_gamma.txt", transpose(layer.γ), ' ')
        writedlm("cppweights_" * RUN_NAME * "/y_beta.txt", transpose(layer.β), ' ')
        writedlm("cppweights_" * RUN_NAME * "/y_mu.txt", transpose(layer.μ), ' ')
        writedlm("cppweights_" * RUN_NAME * "/y_batchnorm_denom.txt", transpose(sqrt.(layer.σ² .+ 1e-6)), ' ')
    elseif j < tuple_len(y_net.layers)
        writedlm("cppweights_" * RUN_NAME * "/y_w" * "$(j-1)" * ".txt", transpose(params[1]), ' ')
        writedlm("cppweights_" * RUN_NAME * "/y_b" * "$(j-1)" * ".txt", transpose(params[2]), ' ')
    else
        writedlm("cppweights_" * RUN_NAME * "/y_w" * "$(j-1)" * ".txt", transpose(params[1]), ' ')
    end
end
