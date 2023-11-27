function main_3dim()
    
    % Path to the file to record the value function and the optimal policy
    % matrix at each interval
    record_file_policy = "/project/Call_Center_Control/analyses_final/matlab/3dim/policy/";
    record_file_value = "/project/Call_Center_Control/analyses_final/matlab/3dim/value/";
    
    % Path to the configuration file
    jsonFilePath = '/project/Call_Center_Control/analyses_final/matlab/3dim_new/config_3dim.json';
    
    % Read the content of the file into a string
    jsonString = fileread(jsonFilePath);

    % Parse the JSON string into MATLAB data
    configData = jsondecode(jsonString);

    % Configuration and Initialization
    config = defineConfig(configData);
    data = readAndPreprocessData(config, configData);
    
    % Run CTMC code
    V = runCTMC(config, data, record_file_policy, record_file_value);

end

function config = defineConfig(configData)
    % Define all constants, file paths, and parameters
    config.class_no = 3; % this code is defined for solving CTMC for 3 dimensional problems
    config.num_policies = factorial(config.class_no); % number of policies we consider
    config.precision = configData.precision; % the precision is set to 5 minutes
    config.total_discretization = configData.total_discretization; % how many time intervals we use to record the policy
    config.hour_total = configData.hour_total; % the length of the time horizon in hours
    
    MINS_IN_HOUR = 60;
    config.total_time = configData.hour_total * MINS_IN_HOUR/ config.precision; % the length of the time horizon in terms of the precision
    
    config.overtimeCost = configData.overtimeCost; % overtime cost rate
    config.upperBounds = configData.upperBounds; % upper bounds of the state space
    config.numIntervals = configData.numIntervals; % number of intervals to discretize the time horizon for the solution of the problem % we consider 0.1 second intervals
    config.deltaT = configData.hour_total / config.numIntervals; % length of the time horizons % we consider 0.1 second intervals
    config.denominator = configData.denominator; % denominator we use to record the policy (determines for which time intervals we record the policy)
    config.num_states = config.upperBounds(1) * config.upperBounds(2) * config.upperBounds(3); % the dimension of the state space
end
 
function data = readAndPreprocessData(config, configData)
    % Read and preprocess data
    data.lambdaClass1 = readmatrix(configData.lambda_class1_path); % arrival rates for class 1 (per 5 minute) 
    data.lambdaClass2 = readmatrix(configData.lambda_class2_path); % arrival rates for class 2 (per 5 minute)
    data.lambdaClass3 = readmatrix(configData.lambda_class3_path); % arrival rates for class 3 (per 5 minute)
    
    data.agents = readmatrix(configData.agents_path); % number of agents at every interval
    data.mu = readmatrix(configData.mu_path); % hourly service rates
    data.theta = readmatrix(configData.theta_path); % hourly abandonment rates
    data.cost = readmatrix(configData.cost_path); % hourly cost rates

    repetitions = config.total_discretization/ config.total_time; % we have to repeat the data based on how we discretize the time

    data.lambdaClass1 = repelem(data.lambdaClass1, repetitions); 
    data.lambdaClass2 = repelem(data.lambdaClass2, repetitions);
    data.lambdaClass3 = repelem(data.lambdaClass3, repetitions);

    data.agents = repelem(data.agents, repetitions);

    MINS_IN_HOUR = 60;
    CONVERSION_CONSTANT = 12; % conversion factor to convert 5 minute into hours

    data.lambdaClass1 = data.lambdaClass1 * CONVERSION_CONSTANT;
    data.lambdaClass2 = data.lambdaClass2 * CONVERSION_CONSTANT;
    data.lambdaClass3 = data.lambdaClass3 * CONVERSION_CONSTANT;

    %scale the system parameters with deltaT
    %converting the units of the data based on the precision specified
    data.mu_scaled = data.mu * config.deltaT; % mu scaled with deltaT
    data.theta_scaled = data.theta * config.deltaT; % theta scaled with deltaT
    data.cost_scaled = data.cost * config.deltaT; % cost scaled with deltaT
    data.lambdaClass1_scaled = data.lambdaClass1 * config.deltaT; % lambdaClass1 scaled with deltaT
    data.lambdaClass2_scaled = data.lambdaClass2 * config.deltaT; % lambdaClass2 scaled with deltaT
    data.lambdaClass3_scaled = data.lambdaClass3 * config.deltaT; % lambdaClass2 scaled with deltaT
end

function [u_1, u_2, u_3] = calculatePolicyUtilities(policy, x1, x2, x3, agents)
    % calculatePolicyUtilities Computes utility values based on a specified policy.
    %
    % This function calculates the utility values for a given policy by adjusting the
    % order of the state parameters (x1, x2, x3) as per the policy's requirement.
    %
    % Inputs:
    %   policy - Integer representing the policy number (1 to 6)
    %   x1, x2, x3 - State parameters for the policy calculation
    %   agents - Number of agents
    %
    % Outputs:
    %   u_1, u_2, u_3 - Utility values corresponding to the policy

    switch policy
        case 1 % Policy 1 - r1 > r2 > r3
            u_1 = max(x1 - agents, 0);
            u_2 = max(x2 - max(agents - x1, 0), 0);
            u_3 = max(x3 - max(max(agents - x1, 0) - x2, 0), 0);
        
        case 2 % Policy 2 - r1 > r3 > r2
            u_1 = max(x1 - agents, 0);
            u_3 = max(x3 - max(agents - x1, 0), 0);
            u_2 = max(x2 - max(max(agents - x1, 0) - x3, 0), 0);
        
        case 3 % Policy 3 - r2 > r1 > r3
            u_2 = max(x2 - agents,0);
            u_1 = max(x1 - max(agents - x2, 0), 0);
            u_3 = max(x3 - max(max(agents - x2, 0) - x1 , 0),0);
        
        case 4 % Policy 4 - r2 > r3 > r1
            u_2 = max(x2 - agents,0);
            u_3 = max(x3 - max(agents - x2, 0), 0);
            u_1 = max(x1 - max(max(agents - x2, 0) - x3 , 0),0);
        
        case 5 % Policy 5 - r3 > r1 > r2
            u_3 = max(x3 - agents,0);
            u_1 = max(x1 - max(agents - x3, 0), 0);
            u_2 = max(x2 - max(max(agents - x3, 0) - x1 , 0),0);
        
        case 6 % Policy 6 - r3 > r2 > r1
            u_3 = max(x3 - agents,0);
            u_2 = max(x2 - max(agents - x3, 0), 0);
            u_1 = max(x1 - max(max(agents - x3, 0) - x2 , 0),0);
        
        otherwise
            error('Invalid policy number');
    end

end

function pol_value = calculatePolicyValue(data, res_minus1, res_minus2,  res_minus3, u_1, u_2, u_3)
    % Calculate and return policy value
    pol_value = (data.cost_scaled(1) + (data.mu_scaled(1) - data.theta_scaled(1)).*res_minus1).*u_1 + (data.cost_scaled(2) + (data.mu_scaled(2) - data.theta_scaled(2)).*res_minus2).*u_2 + (data.cost_scaled(3) + (data.mu_scaled(3) - data.theta_scaled(3)).*res_minus3).*u_3;
end


function diff_minus = calculateDeltaMinus(V, upperBounds)
    % Calculate the delta_minus tensor for given V and upper bounds.
    % Inputs:
    %   V - The value matrix
    %   upperBounds - Upper bounds for the dimensions
    % Outputs:
    %   diff_minus - Calculated tensor for delta_minus

    % Initialize tensor for delta_minus
    diff_minus = zeros(upperBounds(1), upperBounds(2), upperBounds(3), 3);
    directions = ["e1", "e2", "e3"];
    
    % Shift V matrix for delta calculations
    for i = 1:3
        diff_minus(:,:,:,i) = V - shiftMatrix(V, upperBounds, directions(i), 'minus');
    end
end

function diff_plus = calculateDeltaPlus(V, upperBounds)
    % Calculate the delta_plus tensor for given V and upper bounds.
    % Inputs:
    %   V - The value matrix
    %   upperBounds - Upper bounds for the dimensions
    % Outputs:
    %   diff_plus - Calculated tensor for delta_plus

    % Initialize tensor for delta_plus
    diff_plus = zeros(upperBounds(1), upperBounds(2), upperBounds(3), 3);
    directions = ["e1", "e2", "e3"];

    % Shift V matrix for delta calculations
    for i = 1:3
        diff_plus(:,:,:,i) = shiftMatrix(V, upperBounds, directions(i), 'plus') - V;
    end
end

function matShifted = shiftMatrix(mat, upperBounds, direction, operation)
    % Shifts the matrix 'mat' based on the direction and operation.
    % Inputs:
    %   mat - The matrix to be shifted
    %   upperBounds - Upper bounds for the dimensions
    %   direction - The direction of the shift ('e1' or 'e2' or 'e3')
    %   operation - The type of operation ('minus' or 'plus')
    % Outputs:
    %   matShifted - The shifted matrix

    matShifted = mat;
    switch direction
        case 'e1'
            if strcmp(operation, 'minus')
                matShifted(2:upperBounds(1), :, :) = mat(1:upperBounds(1)-1, :, :);
            elseif strcmp(operation, 'plus')
                matShifted(1:upperBounds(1)-1, :, :) = mat(2:upperBounds(1), :, :);
            end
        case 'e2'
            if strcmp(operation, 'minus')
                matShifted(:, 2:upperBounds(2), :) = mat(:, 1:upperBounds(2)-1, :);
            elseif strcmp(operation, 'plus')
                matShifted(:, 1:upperBounds(2)-1, :) = mat(:, 2:upperBounds(2), :);
            end
        case 'e3'
            if strcmp(operation, 'minus')
                matShifted(:, :, 2:upperBounds(3)) = mat(:, :, 1:upperBounds(3)-1);
            elseif strcmp(operation, 'plus')
                matShifted(:, :, 1:upperBounds(3)-1) = mat(:, :, 2:upperBounds(3));
            end
        otherwise
            error('Invalid shift direction or operation');
    end
end

function terminal_cost = calculateTerminalCost(config, data)
    %Calculate the overtime cost matrix
    % Inputs:
    %   data - Struct containing lambdaClass1_scaled, lambdaClass2_scaled, etc.
    %   config - Configuration parameters like denominator, mu_scaled, etc.
    %
    % Outputs:    
    %   terminal_cost - Matrix representing the calculated overtime costs

    last_interval = config.total_discretization;
    
    % Vectorized computation for overtime cost matrix
    [iGrid, jGrid, kGrid] = ndgrid(1:config.upperBounds(1), 1:config.upperBounds(2), 1:config.upperBounds(3));
    %total people in queue overtime
    totalPeople = (iGrid + jGrid + kGrid - 3) - data.agents(last_interval);
    totalPeople(totalPeople < 0) = 0; % Replace negative values with zero
    terminal_cost = config.overtimeCost * totalPeople;

end

function [priority_rule, change] = hjb(interval, data, config, V)
    % HJB Function: Computes the Hamilton-Jacobi-Bellman (HJB) change for a given state.
    
    % Inputs:
    %   interval - Current time interval
    %   data - Struct containing lambdaClass1_scaled, lambdaClass2_scaled, etc.
    %   config - Configuration parameters like denominator, mu_scaled, etc.
    %   V - Value function
    %
    % Outputs:
    %   change - Computed HJB change for the given state.
    %   priority rule - Computed optimal priority rule
    
    data_interval = ceil(interval/config.denominator);
    arrivals_scaled = [data.lambdaClass1_scaled(data_interval), data.lambdaClass2_scaled(data_interval), data.lambdaClass3_scaled(data_interval)];
    agents_count = data.agents(data_interval);
    
    % generating the state space
    x_axis = config.upperBounds(1) - 1;
    y_axis = config.upperBounds(2) - 1;
    z_axis = config.upperBounds(3) - 1;

    [x1, x2, x3] = ndgrid(0:x_axis,0:y_axis,0:z_axis); % state space

    res_minus = calculateDeltaMinus(V, config.upperBounds);
    res_minus1 = res_minus(:,:,:,1); % class 1 minus_delta
    res_minus2 = res_minus(:,:,:,2); % class 2 minus_delta
    res_minus3 = res_minus(:,:,:,3); % class 3 minus_delta

    res_plus = calculateDeltaPlus(V, config.upperBounds);
    res_plus1 = res_plus(:,:,:,1); % class 1 plus_delta
    res_plus2 = res_plus(:,:,:,2); % class 2 plus_delta
    res_plus3 = res_plus(:,:,:,3); % class 3 plus_delta

    pol_values_concatenated = zeros(size(res_minus1, 1), size(res_minus2, 2), size(res_minus3, 3), config.num_policies);

    for policy = 1:config.num_policies
        [u_1, u_2, u_3] = calculatePolicyUtilities(policy, x1, x2, x3, agents_count);

        % Compute policy value using the utility values
        pol_values_concatenated(:, :, :, policy) =  calculatePolicyValue(data, res_minus1, res_minus2,  res_minus3, u_1, u_2, u_3);
    end
    
    % Compute terms of the HJB equation
    ft = - arrivals_scaled(1).*res_plus1 - arrivals_scaled(2).*res_plus2 - arrivals_scaled(3).*res_plus3;
    st = data.mu_scaled(1).*x1.*res_minus1 + data.mu_scaled(2).*x2.*res_minus2 + data.mu_scaled(3).*x3.*res_minus3;
    [tt_min, opt_policy_ind] = min(pol_values_concatenated, [], 4);
    tt = -1 * tt_min;
    
    % Determine the priority rule
    priority_rule = opt_policy_ind;
    change = ft + st + tt;
end



function V = runCTMC(config, data, record_file_policy, record_file_value)

    % solve the associated CTMC and return the final value matrix "V"

    % initialize
    
    interval = config.numIntervals;
 
    V = calculateTerminalCost(config, data); % setting it to terminal cost (g function)

    while interval > 0
        
        % hjb matrix
        
        [priority_rule, hjb_tens] = hjb(interval, data, config, V);
        
        policy_record_interval = interval/config.denominator;

        if (rem(policy_record_interval,1) == 0) 
            reshaped_priority_rule = reshape(priority_rule, [config.num_states,1]);
            txt_policy = sprintf('%spolicy%i.txt', record_file_policy, policy_record_interval);
            writetable(array2table(reshaped_priority_rule), txt_policy, 'WriteVariableNames',0) %writing policy
        end

        V = V - hjb_tens; %update the value function %picard iteration

        if (interval == 1)
            reshaped_V = reshape(V, [config.num_states,1]);
            txt_value = sprintf('%svalue%i.txt', record_file_value, interval - 1);
            writetable(array2table(reshaped_V), txt_value) %writing value function
        end

        interval = interval - 1; % update the interval
        disp(interval)
        
    end
end
