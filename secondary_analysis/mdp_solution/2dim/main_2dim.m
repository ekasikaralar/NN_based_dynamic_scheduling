function main_2dim()
    
    % Path to the file to record the value function and the optimal policy
    % matrix at each interval
    record_file_policy = "/project/Call_Center_Control/analyses_final/matlab/2dim/policy/";
    record_file_value = "/project/Call_Center_Control/analyses_final/matlab/2dim/value/";
    
    % Path to the configuration file
    jsonFilePath = '/project/Call_Center_Control/analyses_final/matlab/2dim/config_2dim.json';
    
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
    config.class_no = 2; % this code is defined for solving CTMC for 2 dimensional problems
    config.precision = configData.precision; % the precision is set to 5 minutes
    config.total_discretization = configData.total_discretization; % how many time intervals we use to record the policy
    config.hour_total = configData.hour_total; % the length of the time horizon in hours
    
    MINS_IN_HOUR = 60;
    config.total_time = configData.hour_total * MINS_IN_HOUR / config.precision; % the length of the time horizon in terms of the precision
    
    config.overtimeCost = configData.overtimeCost; % overtime cost rate
    config.upperBounds = configData.upperBounds; % upper bounds of the state space
    config.numIntervals = configData.numIntervals; % number of intervals to discretize the time horizon
    config.deltaT = configData.hour_total / config.numIntervals; % scaling factor
    config.denominator = configData.denominator; % denominator we use to record the policy
    config.num_states = config.upperBounds(1) * config.upperBounds(2); % the dimension of the state space
end

function data = readAndPreprocessData(config, configData)
    % Read and preprocess data
    data.lambdaClass1 = readmatrix(configData.lambda_class1_path); % arrival rates for class 1 (per 5 minute) 
    data.lambdaClass2 = readmatrix(configData.lambda_class2_path); % arrival rates for class 2 (per 5 minute)
    data.agents = readmatrix(configData.agents_path); % number of agents at every interval
    data.mu = readmatrix(configData.mu_path); % hourly service rates
    data.theta = readmatrix(configData.theta_path); % hourly abandonment rates
    data.cost = readmatrix(configData.cost_path); % hourly cost rates

    repetitions = config.total_discretization/ config.total_time; % we have to repeat the data based on how we discretize the time

    data.lambdaClass1 = repelem(data.lambdaClass1, repetitions); 
    data.lambdaClass2 = repelem(data.lambdaClass2, repetitions);
    data.agents = repelem(data.agents, repetitions);

    MINS_IN_HOUR = 60;
    CONVERSION_CONSTANT = 12; % conversion factor to convert 5 minute into hours

    data.lambdaClass1 = data.lambdaClass1 * CONVERSION_CONSTANT;
    data.lambdaClass2 = data.lambdaClass2 * CONVERSION_CONSTANT;

    %scale the system parameters with deltaT
    data.mu_scaled = data.mu * config.deltaT; % mu scaled with deltaT
    data.theta_scaled = data.theta * config.deltaT; % theta scaled with deltaT
    data.cost_scaled = data.cost * config.deltaT; % cost scaled with deltaT
    data.lambdaClass1_scaled = data.lambdaClass1 * config.deltaT; % lambdaClass1 scaled with deltaT
    data.lambdaClass2_scaled = data.lambdaClass2 * config.deltaT; % lambdaClass2 scaled with deltaT
   
end


function priority_rule = determinePriorityRule(pol1_value, pol2_value)
    % Determine and return the priority rule based on policy values
    
    priority_rule = (pol1_value < pol2_value) + (pol1_value == pol2_value)*2; 
    
    %1 indices would show the class 1 is prioritized
    %0 indices would show the class 2 is prioritized
    %2 indices would show where both policies give the same value (when there are enough servers to serve both classes of calls)
end

function [u_1_pol1, u_2_pol1, u_1_pol2, u_2_pol2] = computePolicyUtilityIndices(x1, x2, agents_count)
    % Compute and return policy utility indices
    u_1_pol1 = max(x1 - agents_count,0); %first index of policy 1 
    u_2_pol1 = max(x2 - max(agents_count - x1,0),0); %second index of policy 1 
    u_1_pol2 = max(x1 - max(agents_count - x2,0),0); %first index of policy 2  
    u_2_pol2 = max(x2 - agents_count,0); %second index of policy 2 
end


function pol_value = calculatePolicyValue(data, res_minus1, res_minus2, u_1, u_2)
    % Calculate and return policy value
    pol_value = (data.cost_scaled(1) + (data.mu_scaled(1) - data.theta_scaled(1)).*res_minus1).*u_1 + (data.cost_scaled(2) + (data.mu_scaled(2) - data.theta_scaled(2)).*res_minus2).*u_2;
end


function diff_minus = calculateDeltaMinus(V, upperBounds)
    % Calculate the delta_minus tensor for given V and upper bounds.
    % Inputs:
    %   V - The value matrix
    %   upperBounds - Upper bounds for the dimensions
    % Outputs:
    %   diff_minus - Calculated tensor for delta_minus

    % Initialize tensor for delta_minus
    diff_minus = zeros(upperBounds(1), upperBounds(2), 2);

    % Shift V matrix for delta calculations
    matShiftedE1 = shiftMatrix(V, upperBounds, 'e1', 'minus');
    matShiftedE2 = shiftMatrix(V, upperBounds, 'e2', 'minus');

    % Compute delta_minus for both classes
    diff_minus(:,:,1) = V - matShiftedE1; % V - V_{e1} for class 1
    diff_minus(:,:,2) = V - matShiftedE2; % V - V_{e2} for class 2
end

function diff_plus = calculateDeltaPlus(V, upperBounds)
    % Calculate the delta_plus tensor for given V and upper bounds.
    % Inputs:
    %   V - The value matrix
    %   upperBounds - Upper bounds for the dimensions
    % Outputs:
    %   diff_plus - Calculated tensor for delta_plus

    % Initialize tensor for delta_plus
    diff_plus = zeros(upperBounds(1), upperBounds(2), 2);

    % Shift V matrix for delta calculations
    matShiftedE1 = shiftMatrix(V, upperBounds, 'e1', 'plus');
    matShiftedE2 = shiftMatrix(V, upperBounds, 'e2', 'plus');

    % Compute delta_plus for both classes
    diff_plus(:,:,1) = matShiftedE1 - V; % V_{e1} - V for class 1
    diff_plus(:,:,2) = matShiftedE2 - V; % V_{e2} - V for class 2
end

function matShifted = shiftMatrix(mat, upperBounds, direction, operation)
    % Shifts the matrix 'mat' based on the direction and operation.
    % Inputs:
    %   mat - The matrix to be shifted
    %   upperBounds - Upper bounds for the dimensions
    %   direction - The direction of the shift ('e1' or 'e2')
    %   operation - The type of operation ('minus' or 'plus')
    % Outputs:
    %   matShifted - The shifted matrix

    matShifted = mat;
    switch direction
        case 'e1'
            if strcmp(operation, 'minus')
                matShifted(2:upperBounds(1), :) = mat(1:upperBounds(1)-1, :);
            elseif strcmp(operation, 'plus')
                matShifted(1:upperBounds(1)-1, :) = mat(2:upperBounds(1), :);
            end
        case 'e2'
            if strcmp(operation, 'minus')
                matShifted(:, 2:upperBounds(2)) = mat(:, 1:upperBounds(2)-1);
            elseif strcmp(operation, 'plus')
                matShifted(:, 1:upperBounds(2)-1) = mat(:, 2:upperBounds(2));
            end
        otherwise
            error('Invalid shift direction or operation');
    end
end

function [priority_rule, change] = hjb(interval, data, config, V)
   
    % HJB Function: Computes the Hamilton-Jacobi-Bellman (HJB) change for a given state.
    
    % Inputs:
    %   interval - Current time interval
    %   data - Struct containing lambdaClass1_scaled, lambdaClass2_scaled, etc.
    %   config - Configuration parameters like denominator, mu_scaled, etc.
    %
    % Outputs:
    %   change - Computed HJB change for the given state.
    %   priority rule - Computed optimal priority rule
    
    data_interval = ceil(interval/config.denominator);
    arrivals_scaled = [data.lambdaClass1_scaled(data_interval), data.lambdaClass2_scaled(data_interval)];
    
    % generating the state matrix
    nrows = config.upperBounds(1) - 1;
    ncols = config.upperBounds(2) - 1;

    [x1, x2] = ndgrid(0:nrows,0:ncols); % state space

    res_minus  = calculateDeltaMinus(V, config.upperBounds);
    res_minus1 = res_minus(:,:,1); %class 1 minus_delta
    res_minus2 = res_minus(:,:,2); %class 2 minus_delta

    res_plus = calculateDeltaPlus(V, config.upperBounds);
    res_plus1 = res_plus(:,:,1); %class 1 plus_delta
    res_plus2 = res_plus(:,:,2); %class 2 plus_delta
    
    agents_count = data.agents(data_interval);
    
    [u_1_pol1, u_2_pol1, u_1_pol2, u_2_pol2] = computePolicyUtilityIndices(x1, x2, agents_count);
    
    % Policy values
    % policy 1 is when class 1 is prioritized
    % policy 2 is when class 2 is prioritized
    pol1_value = calculatePolicyValue(data, res_minus1, res_minus2, u_1_pol1, u_2_pol1);
    pol2_value = calculatePolicyValue(data, res_minus1, res_minus2, u_1_pol2, u_2_pol2);

    % Compute terms of the HJB equation
    ft = -arrivals_scaled(1) .* res_plus1 - arrivals_scaled(2) .* res_plus2; % First term
    st = data.mu_scaled(1) .* x1 .* res_minus1 + data.mu_scaled(2) .* x2 .* res_minus2; % Second term
    tt = -min(pol1_value, pol2_value); % Third term
  
    % Determine the priority rule
    priority_rule = determinePriorityRule(pol1_value, pol2_value);
    %1 indices would show the class 1 is prioritized
    %0 indices would show the class 2 is prioritized
    %2 indices would show where both policies give the same value
    
    change = ft + st + tt;
    
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
    [iGrid, jGrid] = ndgrid(1:config.upperBounds(1), 1:config.upperBounds(2));
    %total people in queue overtime
    totalPeople = (iGrid + jGrid - 2) - data.agents(last_interval);
    totalPeople(totalPeople < 0) = 0; % Replace negative values with zero
    terminal_cost = config.overtimeCost * totalPeople;

end


function V = runCTMC(config, data, record_file_policy, record_file_value)

    % solve the associated CTMC and return the final value matrix "V"

    % initialize
    
    interval = config.numIntervals;
 
    V = calculateTerminalCost(config, data); % setting it to terminal cost (g function)

    while interval > 0
        
        % hjb matrix
        
        [priority_rule, hjb_mat] = hjb(interval, data, config, V);
        

        policy_record_interval = interval/config.denominator;

        if (rem(policy_record_interval,1) == 0) 
            txt_policy = sprintf('%spolicy%i.csv', record_file_policy, policy_record_interval);
            writetable(array2table(priority_rule), txt_policy, 'WriteVariableNames',0) %writing policy
        end

        V = V - hjb_mat; %update the value function %picard iteration

        if (interval == 1)
            txt_value = sprintf('%svalue%i.csv', record_file_value, interval - 1);
            writetable(array2table(V), txt_value) %writing value function
        end

        interval = interval - 1; % update the interval
        disp(interval)
    end
end
