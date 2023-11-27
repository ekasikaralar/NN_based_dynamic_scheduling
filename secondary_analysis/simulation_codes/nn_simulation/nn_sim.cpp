#include "nn_sim.h"

namespace simulation {

	//initialization 
	Simulation::Simulation(const std::string& jsonFileName){
		
	// Create a JSON object
    	nlohmann::json config;

	// Open the JSON file
    	std::ifstream file(jsonFileName);
		if (!file.is_open()) {
        	// Handle error - file not found
        	throw std::runtime_error("Unable to open config.json");
    	}

    	// Try to parse the JSON file
    	try {
        	file >> config;
    	} catch (const std::exception& e) {
        	// Handle parsing error
        	throw std::runtime_error("JSON parsing error: " + std::string(e.what()));
    	}

	// Close the JSON file
    	file.close();

		// Accessing configuration values
    		class_no = config["class_no"]; //the dimension of the problem
    		num_interval = config["num_interval"]; //the number of intervals we consider for neural networks
		num_iterations = config["num_iterations"];
		decision_freq = config["decision_freq"]; //decision frequency in seconds
		scaling_factor = config["scaling_factor"]; //scaling factor to convert prelimit state to limiting state

		std::string lambda_path = config["lambda_path"];
		std::string agents_path = config["agents_path"];
		std::string mu_hourly_path = config["mu_hourly_path"];
		std::string theta_hourly_path = config["theta_hourly_path"];
		std::string arr_cdf_path = config["arr_cdf_path"];
		std::string holding_cost_rate_path = config["holding_cost_rate_path"];
		std::string abandonment_cost_rate_path = config["abandonment_cost_rate_path"];
		std::string cost_rate_path = config["cost_rate_path"];
		std::string initialization_path = config["initialization_path"];
		std::string lambda_limit_path = config["lambda_limit_path"];
		neural_network_folder_name = config["nn_folder_name"];//the name of the folder that has the neural network weights

		lambda = readVectorFromCSV(lambda_path);  //per 5 minute arrival rates
		std::vector<double> agents = readVectorFromCSV(agents_path); //number of agents N(t)
		no_server.resize(agents.size());
    		std::transform(agents.begin(), agents.end(), no_server.begin(), 
                   [](double val) { return static_cast<int>(val); });

		mu_hourly = readVectorFromCSV(mu_hourly_path); //hourly common service rate
		theta_hourly = readVectorFromCSV(theta_hourly_path);  //hourly common abandonment rate

		arr_cdf = readMatrixFromCSV(arr_cdf_path); //cumulative distribution function for the arrivals
		holding_cost_rate = readVectorFromCSV(holding_cost_rate_path); //hourly holding cost rate
		abandonment_cost_rate = readVectorFromCSV(abandonment_cost_rate_path); //hourly abandonment cost rate
		initialization = readVectorFromCSV(initialization_path); //initialization of X(0)
		cost_rate = readVectorFromCSV(cost_rate_path); //hourly cost rate
		lambda_trans = readMatrixFromCSV(lambda_limit_path); //limiting hourly arrival rate

		if (class_no == 30 || class_no == 50 || class_no == 100) {
    			// Resize vectors to the size of class_no, using the first element as the fill value
   	 		mu_hourly.resize(class_no, mu_hourly.empty() ? 0 : mu_hourly[0]);
    			theta_hourly.resize(class_no, theta_hourly.empty() ? 0 : theta_hourly[0]);
		}
	}

	std::vector<std::string> Simulation::splitString(const std::string& input, char delimiter) {
	    std::vector<std::string> tokens;
	    std::string token;
	    std::istringstream tokenStream(input);
	    while (std::getline(tokenStream, token, delimiter)) {
	        tokens.push_back(token);
		}
		return tokens;
	}

	// Function to read matrix from CSV file
	std::vector<std::vector<double> > Simulation::readMatrixFromCSV(const std::string& filename) {
	    std::vector<std::vector<double> > matrix;

	    std::ifstream file(filename);
	    if (!file.is_open()) {
	        std::cerr << "Failed to open the file: " << filename << std::endl;
	        return matrix;
	    }

	    std::string line;
	    while (std::getline(file, line)) {
	        std::vector<std::string> row = splitString(line, ',');
	        std::vector<double> matrixRow;
	        for (const std::string& str : row) {
	            matrixRow.push_back(std::stod(str));
	        }
	        matrix.push_back(matrixRow);
	    }

	    file.close();

	    return matrix;
	}
	
	// Function to read vector from CSV file
	std::vector<double> Simulation::readVectorFromCSV(const std::string& filename) {
	    std::vector<double> vec;
	    std::ifstream file(filename);
	    if (!file.is_open()) {
	        std::cout << "Failed to open the file." << std::endl;
	        return vec;
	    }

	    std::string line;
	    while (std::getline(file, line)) {
	        std::stringstream ss(line);
	        std::string cell;
	        while (std::getline(ss, cell, ',')) {
	            double value = std::stod(cell);
	            vec.push_back(value);
	        }
	    }

	    file.close();
	    return vec;
	}
	
	Simulation::~Simulation(){
		std::cout << "Done" << std::endl;
	}

	int Simulation::save(const std::string& record_file){
        
		const char *path_pol = &record_file[0];
		const int myfile_pol = open(path_pol, O_CREAT | O_WRONLY);

		if (myfile_pol != -1){			

			#pragma omp parallel for num_threads(100) // Adjust the number of threads as needed
			for (int iter = 0;  iter < num_iterations; iter++){
		
				std::vector<double> cost; //to save the costs
				cost.assign(class_no + 1, 0);
				std::cout << "Iteration: " << iter << std::endl;

				// Execute simulation
				simulation::Execute exec(class_no, arr_cdf, lambda, mu_hourly, theta_hourly, no_server,
          									holding_cost_rate, abandonment_cost_rate, cost_rate, lambda_trans, num_interval, decision_freq, iter, scaling_factor); 

				cost = exec.run(initialization, neural_network_folder_name);

				std::string results;
				results += std::to_string(iter);
				results += ",";

				for (int j = 0; j < class_no; j++){
					results += std::to_string(cost[j]);
					results += ",";
				}

				results += std::to_string(cost[class_no]); //total_cost
				results += "\n";

				const char *char_results = const_cast<char*>(results.c_str());
				write(myfile_pol, char_results, results.length());
			}
		}

		close(myfile_pol);
		return 0;
	}

	MyNetwork::~MyNetwork()
	{
	};

	// Adds two matrices element-wise. Throws an exception if their dimensions don't match.
	std::vector<std::vector<float> > MyNetwork::add(const std::vector<std::vector<float> >& x, std::vector<float>& y) {
	    // Check for dimension mismatch
		if (x[0].size() != y.size()) {
	        throw "Dimension mismatch";
	    }
	    std::vector<std::vector<float> > result(x.size(), std::vector<float>(x[0].size(), 0));
	    for (int i = 0; i < x.size(); i++) {
	        for (int j = 0; j < x[0].size(); j++) {
	            result[i][j] = x[i][j] + y[j];
	        }
	    }
	    return result;
	}
	
	// Multiplies two matrices. Throws an exception if the number of columns in the first matrix
    // doesn't match the number of rows in the second.
	std::vector<std::vector<float> > MyNetwork::matmul(const std::vector<std::vector<float> >& mat1, const std::vector<std::vector<float> >& mat2) {
	    // Check for dimension mismatch
		if (mat1[0].size() != mat2.size()) {
	        throw "Dimension mismatch";
	    }
	    std::vector<std::vector<float> > result(mat1.size(), std::vector<float>(mat2[0].size(), 0));
	    for (int i = 0; i < mat1.size(); i++) {
	        for (int j = 0; j < mat2[0].size(); j++) {
	            for (int k = 0; k < mat1[0].size(); k++) {
	                result[i][j] += mat1[i][k] * mat2[k][j];
	            }
	        }
	    }
	    return result;
	}

	// Divides each element of a matrix by a corresponding element in a vector.
    // Throws an exception if their dimensions don't match.
	std::vector<std::vector<float> > MyNetwork::divide(const std::vector<std::vector<float> >& x, std::vector<float>& y) {
	    // Check for dimension mismatch
		if (x[0].size() != y.size()) {
	        throw "Dimension mismatch";
	    }
	    std::vector<std::vector<float> > result(x.size(), std::vector<float>(x[0].size(), 0));
	    for (int i = 0; i < x.size(); i++) {
	        for (int j = 0; j < x[0].size(); j++) {
	            result[i][j] = x[i][j] / y[j];
	        }
	    }
	    return result;
	}

	// Subtracts each element of a vector from a corresponding element in a matrix.
    // Throws an exception if their dimensions don't match.
	std::vector<std::vector<float> > MyNetwork::subtract(const std::vector<std::vector<float> >& x, std::vector<float>& y) {
	    // Check for dimension mismatch
		if (x[0].size() != y.size()) {
	        throw "Dimension mismatch";
	    }
	    std::vector<std::vector<float> > result(x.size(), std::vector<float>(x[0].size(), 0));
	    for (int i = 0; i < x.size(); i++) {
	        for (int j = 0; j < x[0].size(); j++) {
	            result[i][j] = x[i][j] - y[j];
	        }
	    }
	    return result;
	}

	// Multiplies each element of a matrix by a corresponding element in a vector.
    // Throws an exception if their dimensions don't match.
	std::vector<std::vector<float> > MyNetwork::multiply(const std::vector<std::vector<float> >& x, std::vector<float>& y) {
	    if (x[0].size() != y.size()) {
	        throw "Dimension mismatch";
	    }
	    std::vector<std::vector<float> > result(x.size(), std::vector<float>(x[0].size(), 0));
	    for (int i = 0; i < x.size(); i++) {
	        for (int j = 0; j < x[0].size(); j++) {
	            result[i][j] = x[i][j] * y[j];
	        }
	    }
	    return result;
	}

	// Applies the Leaky ReLU function with an alpha of 0.2 to each element of a matrix.
	std::vector<std::vector<float> > MyNetwork::leaky_relu(const std::vector<std::vector<float> >& x) {
	    std::vector<std::vector<float> > result(x.size(), std::vector<float>(x[0].size(), 0));
	    for (int i = 0; i < x.size(); i++) {
	        for (int j = 0; j < x[0].size(); j++) {
	            result[i][j] = x[i][j] > 0 ? x[i][j] : 0.2*x[i][j];
	        }
	    }
	    return result;
	}

	// Function to read matrix from CSV or txt file
	std::vector<std::vector<float> > MyNetwork::readMatrix(const std::string &filename) {
		std::vector<std::vector<float> > result;
	    std::ifstream file(filename);
	    std::string line;
	    while (getline(file, line)) {
	        std::vector<float> row;
	        std::istringstream ss(line);
	        float value;
	        while (ss >> value) {
	            row.push_back(value);
	        }
	        result.push_back(row);
	    }
	    file.close();
		return result;
	}

	// Function to read vector from CSV or txt file
	std::vector<float> MyNetwork::readVector(const std::string &filename) {
		std::vector<float> result;
	    std::ifstream file(filename);
	    std::string line;
	    while (getline(file, line)) {
	        std::vector<float> row;
	        std::istringstream ss(line);
	        float value;
	        while (ss >> value) {
	            row.push_back(value);
	        }
			file.close();
	        return row;
	    }
	}

	/**
 		* Performs the forward pass of the neural network.
 		*
 		* @param x The input matrix to the network.
 		* @return The output matrix after processing through the network layers.
 		*
 		* The function processes the input through several layers of the network, applying 
 		* different operations at each stage:
 		* 1. Batch normalization: The input matrix 'x' is first normalized using subtracting mean 
 		*    (this->mu), dividing by a batch normalization denominator (this->batchnorm_denom), 
 		*    and then scaled and shifted using parameters 'gamma' and 'beta'.
 		* 2. Leaky ReLU activation: After batch normalization, the result is passed through 
 		*    a series of layers, each consisting of a matrix multiplication with a weight 
 		*    matrix (this->w1, w2, w3, w4) followed by adding a bias vector (this->b1, b2, b3, b4) 
 		*    and applying the leaky ReLU activation function.
 		* 3. Final layer: The output from the last Leaky ReLU layer is multiplied by 
	 	*    the final weight matrix (this->w5) to produce the final output of the network.
 	*/
	std::vector<std::vector<float> > MyNetwork::forward(const std::vector<std::vector<float> >& x) {
		std::vector<std::vector<float> > out;
		out = add(multiply(divide(subtract(x, this->mu), this->batchnorm_denom), this->gamma), this->beta);
		out = leaky_relu(add(matmul(out, this->w1), this->b1));
		out = leaky_relu(add(matmul(out, this->w2), this->b2));
		out = leaky_relu(add(matmul(out, this->w3), this->b3));
		out = leaky_relu(add(matmul(out, this->w4), this->b4));
		return matmul(out, this->w5);
	}

	void MyNetwork::load(std::string& neural_network_folder_name, std::string &filename) {
	
		if (filename.empty()) {
        	throw std::invalid_argument("Filename cannot be empty");
    	}	
		// Load network parameters
		this->gamma = readVector(neural_network_folder_name + filename + "_gamma.txt");
		this->beta = readVector(neural_network_folder_name + filename + "_beta.txt");
		this->mu = readVector(neural_network_folder_name + filename + "_mu.txt");
		
		this->batchnorm_denom = readVector(neural_network_folder_name + filename + "_batchnorm_denom.txt");
		
		this->b1 = readVector(neural_network_folder_name + filename + "_b1.txt");
		this->b2 = readVector(neural_network_folder_name + filename + "_b2.txt");
		this->b3 = readVector(neural_network_folder_name + filename + "_b3.txt");
		this->b4 = readVector(neural_network_folder_name + filename + "_b4.txt");
		
		this->w1 = readMatrix(neural_network_folder_name + filename + "_w1.txt");
		this->w2 = readMatrix(neural_network_folder_name + filename + "_w2.txt");
		this->w3 = readMatrix(neural_network_folder_name + filename + "_w3.txt");
		this->w4 = readMatrix(neural_network_folder_name + filename + "_w4.txt");
		this->w5 = readMatrix(neural_network_folder_name + filename + "_w5.txt");
	}

	Execute::Execute(int& class_no_,
							std::vector<std::vector<double>>& arr_cdf_,
							std::vector<double>& lambda_,
							std::vector<double>& mu_hourly_,
							std::vector<double>& theta_hourly_, 
							std::vector<int>& no_server_,
							std::vector<double>& holding_cost_rate_,
							std::vector<double>& abandonment_cost_rate_,
							std::vector<double>& cost_rate_,
							std::vector<std::vector<double>>& lambda_trans_,
							int& num_interval_,
							int& decision_freq_,
							int& seed,
							double& scaling_factor_)
	{
		class_no = class_no_;

		arr_cdf = arr_cdf_; 
		lambda = lambda_;
		no_server = no_server_; 

		mu_hourly = mu_hourly_; 
		theta_hourly = theta_hourly_;

		holding_cost_rate = holding_cost_rate_;
		abandonment_cost_rate = abandonment_cost_rate_;
		cost_rate = cost_rate_;
		lambda_trans = lambda_trans_;
		num_interval = num_interval_;
		decision_freq = decision_freq_;
        scaling_factor = scaling_factor_;

      	generator.seed(seed);
		queue_init();
	}

	Execute::~Execute()
	{
		delete[] nn_zs;
	}

	//Initializes the queues for each class in the simulation
	void Execute::queue_init()
	{
		
		// Initialize 'not_an_empty_queue' with 'inf'.
		std::vector<double> not_an_empty_queue{inf};

    	// Reserve space for efficiency if 'class_no' is large.
    	queue_list.reserve(class_no);
    	arr_list.reserve(class_no);
    	abandonment_list.reserve(class_no);

    	// Initialize the queues for each class.
    	for (int i = 0; i < class_no; i++) {
       		queue_list.push_back(empty_queue);    // Add a copy of 'empty_queue' to 'queue_list'.
        	arr_list.push_back(empty_queue);      // Add a copy of 'empty_queue' to 'arr_list'.
        	abandonment_list.push_back(not_an_empty_queue); // Add a copy of 'not_an_empty_queue' to 'abandonment_list'.
    	}
		
	}

	/**
 		* Generates a random interarrival time based on an exponential distribution.
 		* 
 		 @param interval The index used to select the arrival rate from the 'lambda' vector.
 		* @return A randomly generated interarrival time.
 	*/
	double Execute::generate_interarrival(int& interval){	
			const double conversionFactor = 12; //number of 5 minutes in an hour
			double arrival_rate = lambda[interval]*conversionFactor; //hourly arrival rate
    		std::exponential_distribution<double> interarrivalTimeDistribution(arrival_rate);
    		return interarrivalTimeDistribution(generator);   
	}

	/**
		* Generates a random abandonment time based on an exponential distribution.
 		* 
 		* @param cls The class index used to select the abandonment rate from 'theta_hourly'.
 		* @return A randomly generated abandonment time.
 	*/
	double Execute::generate_abandon(int& cls) {
		double abandonment_rate = theta_hourly[cls]; //hourly abandonment rate
    std::exponential_distribution<double> abandonmentDistribution(abandonment_rate);
    return abandonmentDistribution(generator);
	}

	/**
 		* Generates a random service time based on an exponential distribution.
 		*
 		* The service time is determined by calculating the service rate as a sumproduct of the number
 		* of services in each class ('num_in_service') and the hourly service rate for each class ('mu_hourly').
 		*
 		* @return A randomly generated service time.
 	*/
	double Execute::generate_service(){
		double service_rate = 0;				
		for (int i = 0; i < class_no; ++i){
			//hourly service rate
			service_rate += num_in_service[i+1] * mu_hourly[i]; //sumproduct of num_in_service and mu_hourly
		}
	    	std::exponential_distribution<double> serviceTimeDistribution(service_rate);
	    	return serviceTimeDistribution(generator);
	}

	/**
 		* Determines the priority order for classes based on their effective holding cost rate.
 		*
 		* This function calculates a priority order by sorting the classes based on their effective holding cost rates. 
 		*
 		* @param num_in_system The number of items in the system for each class 
 		* @param interval The current time interval 
 		* @return A vector of class indices sorted by their priority
 	*/
	std::vector<double> Execute::queueing_discipline(std::vector<int>& num_in_system, int& interval){
		std::string filename;
		//limiting system state
		std::vector<float> scaled_x;
		scaled_x.assign(class_no, 0);
		int priority_ind;

		for (int i = 0; i < class_no; i++){
			scaled_x[i] = (num_in_system[i+1] - scaling_factor * lambda_trans[i][interval]/mu_hourly[i])/sqrt(scaling_factor); //scaled_x
		}

		std::vector<std::vector<float>> input_tensor(1);
		for (int i = 0; i < class_no; i++){
			input_tensor[0].push_back(scaled_x[i]); 
		}
	
		//neural network estimation of the gradient of the value function
		std::vector<float> gradient = load_and_predict(filename, input_tensor);

		std::vector<double>kappa;
		for (int i = 0; i < class_no; i++) {
			//effective holding cost function
			kappa.push_back(-1 * (cost_rate[i] + (mu_hourly[i] - theta_hourly[i]) * gradient[i]));
		}
		
		//prioritizing classes based on their effective holding cost
		std::vector<double> priority_order = argsort(kappa);

		return priority_order;
	}

	std::vector<float> Execute::load_and_predict(std::string &filename, std::vector<std::vector<float>> &input_tensor){
		
		std::vector<std::vector<float>> out;
		int ind;

		int one_min_interval = std::min(int(sim_clock*60),1019);
		int thirty_sec_interval = std::min(int(sim_clock*120),2039);
		int twenty_sec_interval = std::min(int(sim_clock*180),3059);
		int fifteen_sec_interval = std::min(int(sim_clock*240),4079);
		
		std::vector<int> intervals = {interval, one_min_interval, thirty_sec_interval, twenty_sec_interval, fifteen_sec_interval};
		if (decision_freq == 300){ //5 minute decision frequency
			ind = intervals[0];
		} else if (decision_freq == 60){//1 minute decision frequency
			ind = intervals[1];
		} else if (decision_freq == 30){//30 second decision frequency
			ind = intervals[2];
		} else if (decision_freq == 20){//20 second decision frequency
			ind = intervals[3];
		} else if (decision_freq == 15){//15 second decision frequency
			ind = intervals[4];
		}
		
		out = nn_zs[ind].forward(input_tensor);
		
		return out[0];
	}

	/**
 		* Handles the arrival event in a queueing system simulation.
 		* 
 		* This function updates the state of the system upon the arrival of a new entity (person, job, etc.) and decides 
		* whether to serve the new arrival immediately or queue it, based on the current system state and queueing discipline.
 		*
 		* @param interval Current time interval.
 		* @param cls Class of the arriving entity.
 		* @param pre_interval Time interval before the current arrival event.
 		* @param post_interval Time interval after the current arrival event.
 	*/
	void Execute::optimal_policy_calculation(int& interval){
			//optimal policy initialization based on preemptive resume scheduling
			std::vector<int> optimal_policy;
			optimal_policy.assign(class_no,0);

			//we calculate the relative priority of the classes to determine who to send to service and who to keep in the queue
			std::vector<double> priority_order;
			priority_order = queueing_discipline(num_in_system, interval);

			int avail_server_num;
			int num_served = 0;
			int ind;

			//we determine the optimal policy based on the queueing discipline and preemptive scheduling
			for (int i = 0; i < priority_order.size(); i++){
				ind = priority_order[i];
				avail_server_num = std::max(no_server[interval] - num_served, 0);
				optimal_policy[ind] = std::max(num_in_system[ind + 1] - avail_server_num, 0);
				num_served += std::min(num_in_system[ind + 1], avail_server_num);
			}

			//if the difference between people in the queue at the moment and what the optimal policy proposed is positive for any class,
			//we send the people in the amount of the difference to service, and remove them from the queue
			//otherwise some people in service are added to the queue to serve higher priority classes
			int* diff = new int[class_no];

			//determine if the difference is positive or negative
			for (int i = 0; i < class_no; i++){

				diff[i] = num_in_queue[i+1] - optimal_policy[i];
				num_in_service[i + 1] += diff[i];

				num_in_queue[0] -= diff[i];
				num_in_queue[i + 1] -= diff[i];

				//if difference is positive, some people are removed from the queue and send to service
				//next departure event is scheduled
				if (diff[i] >= 0){
					for (int j = 0; j < diff[i]; j++){
						remove_queue(i);
						t_depart = sim_clock + generate_service();
					}
				}
				//if difference is negative, service for some people stops and they are sent back to queue to serve higher priority classes
				else if (diff[i] < 0){
					for (int j = 0; j < abs(diff[i]); j++){
						add_queue(t_event,i);
					}
				}
			}

			//we adjust the number of people in service after all the changes
     	 	num_in_service[0] = 0;
     	 	for (int i = 0; i < class_no; i++){
     	 		num_in_service[0] += num_in_service[i + 1];
     	 	}
			
			delete[] diff;
	}

	/**
 		* Handles the arrival event in a queueing system simulation.
 		* 
 		* This function updates the state of the system upon the arrival of a new entity (person, job, etc.) and decides 
		* whether to serve the new arrival immediately or queue it, based on the current system state and queueing discipline.
 		*
 		* @param interval Current time interval.
 		* @param cls Class of the arriving entity.
 		* @param pre_interval Time interval before the current arrival event.
 		* @param post_interval Time interval after the current arrival event.
 	*/
	void Execute::handle_arrival_event(int& interval, int& cls, int& pre_interval, int& post_interval){ 
		
		//an additional person is added to the system
	    num_in_system[0] += 1; 
	    num_in_system[cls + 1] += 1; 

		//arrival time for the next arrival event is scheduled
		t_arrival = sim_clock + generate_interarrival(interval);

		//an additional person is added to the queue first
		num_in_queue[0] += 1; 
	    num_in_queue[cls+1] += 1;

		//an additional person is added to the queue_list
	    add_queue(t_event,cls);

		//if there are enough servers to serve everyone in the system, then no need to calculate the optimal policy
	    if ((num_in_system[0] <= no_server[interval] and pre_interval == post_interval) or ((num_in_service[0] < no_server[interval]) and pre_interval == post_interval)) {
		    
			if (num_in_service[0] < no_server[interval]) {
			    //without waiting service entrance
				//someone is added to service
			    num_in_service[0] += 1;
			    num_in_service[cls + 1] += 1;

				//we remove the person from the queue as they enter the service
			    num_in_queue[0] -= 1;
			    num_in_queue[cls + 1] -= 1;
				
				remove_queue(cls); //removing the added job from the queue since the job has arrived into the service area
				t_depart = sim_clock + generate_service(); //schedule service completion for the priority job
			
			//if there is no one in the system, then we make sure the next event is not departure
			} else if (num_in_system[0] == 0){
			    t_depart = inf; 
			}
		}
		//if there are not enough servers to serve everyone in the system or if the number of agents have changed due to interval change (pre_interval != post_interval)
		else {
			optimal_policy_calculation(interval);
		}
	}
	
	/**
 		* This function updates the state of the system when an entity (person, job, etc.) departs from service. It adjusts the 
 		* counts of entities in the system and in service, and determines whether to bring new entities into service based on the 
 		* current state and the queueing discipline.
 		*
 		* @param interval Current time interval.
 		* @param cls Class of the departing entity.
 		* @param pre_interval Time interval before the current departure event.
 		* @param post_interval Time interval after the current departure event.
 	*/
	void Execute::handle_depart_event(int& interval, int& cls, int& pre_interval, int& post_interval){ 
		
		//number of people in the system has decreased as someone left the system
		num_in_system[0] -= 1; 
		num_in_system[cls+1] -=1; 
        
		//number of people in service has decreased as someone departed the service
		num_in_service[0] -= 1;
		num_in_service[cls+1] -= 1;
        
		//if there are enough servers to serve everyone in the system, then no need to calculate the optimal policy
		if (num_in_system[0] < no_server[interval] and pre_interval == post_interval){
			//SERVICE ENTRANCE

			if (num_in_queue[0] > 0 && num_in_service[0] < no_server[interval]){
				//first the class with the highest priority enters
			
				//number of people in service increases by one as someone enters service
				num_in_service[0] += 1;
				num_in_service[cls + 1] += 1;

				//number of people in queue decreases by one as someone transitions from the queue to service
				num_in_queue[0] -= 1;
				num_in_queue[cls + 1] -= 1;
				
				remove_queue(cls);
			}

			//if the number of people in the system is positive, we schedule the next departure event as someone enters the service
			if (num_in_system[0] > 0){
				t_depart = sim_clock + generate_service();
			} else {t_depart = inf;} //the next event has to be an arrival if the system is empty	
		}

		else {
			optimal_policy_calculation(interval);
		}	
	}

	/**
		* This function is called when a customer decides to leave the queue before getting the service. It updates the system's
 		* state by decrementing the count of people in the system and in the specific class of the customer who abandoned. 
 
 		* If the condition of system overload or interval change is met, the function calls 'optimal_policy_calculation' 
 		* to recalibrate the service strategy.
 		*
 		* @param interval The current time interval of the simulation.
 		* @param pre_interval The previous time interval before the current state.
 		* @param post_interval The next time interval after the current state.
 	*/
	void Execute::handle_abandon_event(int& interval, int& pre_interval, int& post_interval){
		
		constexpr double MaxTime = std::numeric_limits<double>::max();

		//removing the person who has abandoned from the system
		num_in_system[0] -= 1; //how many people are in the system at time t - one left
		num_in_system[class_abandon + 1] -=1; //how many people out of class i are in the system at time t -- one left
		//increasing the cumulative number of abandons from the list
		
		num_abandons[0] += 1; //how many people have got their service completed so far by time t
		num_abandons[class_abandon + 1] += 1; //how many class i arrivals have been completed by time t 
		//decreasing the number of people from the queue

		//removing the person who has abandoned from the queue
		num_in_queue[0] -= 1;
		num_in_queue[class_abandon + 1] -= 1;
		
		// Remove abandoned customer from queue and abandonment list
		queue_list[class_abandon].erase(queue_list[class_abandon].begin()+cust_abandon);
		abandonment_list[class_abandon].erase(abandonment_list[class_abandon].begin()+cust_abandon+1);

		
		std::vector<double> minAbandonTimes(class_no, MaxTime);
	
		// Determine the next abandonment time
		if (num_in_queue[0] > 0) {
			for (int i = 0; i < class_no; i++) {
				if (abandonment_list[i].size() != 1) {
					minAbandonTimes[i] = *min_element(abandonment_list[i].begin() + 1, abandonment_list[i].end());
				}
			}
		
			double nextAbandonTime = *min_element(minAbandonTimes.begin(), minAbandonTimes.end());
			t_abandon = nextAbandonTime;

			// Find the class and customer for the next abandonment
			for (int i = 0; i < class_no; i++) {
				auto itr = find(abandonment_list[i].begin(), abandonment_list[i].end(), nextAbandonTime);
				if (itr != abandonment_list[i].end()) {
					class_abandon = i;
					cust_abandon = distance(abandonment_list[i].begin(), itr) - 1;
					break;
				}
			}
		} else {
			t_abandon = MaxTime;
		}

		if ((num_in_system[0] > no_server[interval] or pre_interval != post_interval)) {
           optimal_policy_calculation(interval);
		}
	}

	std::vector<double> Execute::argsort(const std::vector<double> &array) {
		std::vector<double> indices(array.size());
	    std::iota(indices.begin(), indices.end(), 0);
	    std::sort(indices.begin(), indices.end(),
	              [&array](int left, int right) -> bool {
	                  // sort indices according to corresponding array element
	                  return array[left] < array[right];
	              });
	
	    return indices;
	}

	/**
 		* Adds a new arrival to the queue and updates the system state for potential future abandonments.
 		*
 		* This function performs several key operations in the context of a queueing system:
 		* 1. Adds the newly arrived customer, identified by their arrival time and class, to the appropriate queue.
 		* 2. Calculates the time at which this customer may abandon the queue, adding this information to the abandonment list.
 		* 3. Scans across all classes to find the next potential abandonment event. This involves finding the minimum 
 		* 	abandonment time from the abandonment lists of all classes.
 		* 4. Updates the system's state to reflect this new potential abandonment, including the time of the next abandonment 
 		*    event (`t_abandon`), and identifying the class and specific customer (`class_abandon` and `cust_abandon`) who might abandon next.
 		*
 		* @param arr_time The arrival time of the customer.
 		* @param cls The class of the customer, which determines their queue.
 	*/
	void Execute::add_queue(double& arr_time, int& cls) {
    	constexpr double MaxTime = std::numeric_limits<double>::max();

    	// Initialize temporary vector to store minimum abandonment times
    	std::vector<double> min_temp(class_no, MaxTime);

    	// Add the arriving person to the queue and abandonment list
    	queue_list[cls].push_back(arr_time);
    	abandonment_list[cls].push_back(arr_time + generate_abandon(cls));

    	// Find the minimum abandonment time from all the queues
    	for (int i = 0; i < class_no; i++) {
        	if (abandonment_list[i].size() != 1) { 
            	min_temp[i] = *min_element(abandonment_list[i].begin(), abandonment_list[i].end());
        	}
    	}

    	// Update the next abandonment time
   		double min_abandon = *min_element(min_temp.begin(), min_temp.end());
    	t_abandon = min_abandon;

    	// Find the class and customer index for the next potential abandonment
    	for (int i = 0; i < class_no; i++) {
        	auto itr = find(abandonment_list[i].begin(), abandonment_list[i].end(), min_abandon);
       		if (itr != abandonment_list[i].end()) {
            	class_abandon = i;
            	cust_abandon = distance(abandonment_list[i].begin(), itr) - 1;
            	break;
        	}
    	}
	}

	/**
 		* Removes a customer from the queue and updates the system's abandonment time.
 		*
		* This function performs the following operations:
 		* 1. Removes the first customer from the queue of the specified class ('cls'). This is typically called after a customer
 		*    has been served or has abandoned the queue.
 		* 2. Simultaneously, it removes the corresponding entry from the abandonment list for that class, maintaining the 
 		*    synchronization between the queue and the abandonment list.
 		* 3. After removal, the function recalculates the next potential abandonment time for the entire system. This is done by 
 		*    finding the minimum abandonment time across all classes and updating 't_abandon' to reflect this time.
 		* 4. It also identifies which class and customer will potentially be the next to abandon, updating 'class_abandon' and 
 		*    'cust_abandon' accordingly.
 		* 
		* If there are no customers left in the queue, 't_abandon' is set to a very large value ('inf'), indicating no imminent 
 		* abandonment events.
 		*
		* @param cls The class index from which the customer is being removed.
 	*/
	void Execute::remove_queue(int& cls){

		constexpr double MaxTime = std::numeric_limits<double>::max();

		// Remove the customer from the queue and abandonment list for the specified class
		queue_list[cls].pop_front();
		abandonment_list[cls].erase(abandonment_list[cls].begin()+1);
		
		// Initialize temporary vector for minimum abandonment times
    	std::vector<double> min_temp(class_no, MaxTime);

    	// Determine the next abandonment time if there are people in the queue
    	if (num_in_queue[0] > 0) {
        	for (int i = 0; i < class_no; i++) {
            	if (abandonment_list[i].size() != 1) { 
                	min_temp[i] = *min_element(abandonment_list[i].begin() + 1, abandonment_list[i].end());
           	 	}
        	}	

        	double min_abandon = *min_element(min_temp.begin(), min_temp.end());
        	t_abandon = min_abandon;

			//find the class of the customer who will abandon next 
			for (int i = 0; i < class_no; i++){
				auto itr = find(abandonment_list[i].begin(), abandonment_list[i].begin()+abandonment_list[i].size(), min_abandon);
				if (itr != abandonment_list[i].begin()+abandonment_list[i].size()){
					class_abandon = i;
					break;
				}
			}
			
			auto cust_itr = find(abandonment_list[class_abandon].begin()+1, abandonment_list[class_abandon].begin()+abandonment_list[class_abandon].size(), min_abandon); 
			cust_abandon = distance(abandonment_list[class_abandon].begin()+1, cust_itr);

		} else {t_abandon = inf;}
	}


	std::vector<double> Execute::run(std::vector<double> initialization, std::string neural_network_folder_name)
	{	

		constexpr double MaxTime = std::numeric_limits<double>::max();
		double overtime_cost = 2.12;

    	int network_size = 0;
		// Set the network_size based on decision_freq

        if (decision_freq == 300) {
            network_size = 204;
        } else if (decision_freq == 60) {
            network_size = 1020;
        } else if (decision_freq == 30) {
            network_size = 2040;
        } else if (decision_freq == 20) {
            network_size = 3060;
        } else if (decision_freq == 15) {
            network_size = 4080;
        }
		
		// Allocate memory for nn_zs
		nn_zs = new simulation::MyNetwork[network_size];
		
		num_in_system.assign(class_no + 1, 0); //0th index is used for the sum
		num_in_service.assign(class_no + 1, 0); //0th index is used for the sum
		
		// Initialize system state
		// initialization -- all servers are busy		
		for (int i = 0; i < class_no; i++){
			num_in_system[i+1] = static_cast<int>(initialization[i]);	
			num_in_service[i+1] = static_cast<int>(initialization[i]);	
		}

		num_in_service[0] = no_server[0]; //total number of people in service is equal to initial number of agents
		num_in_system[0] = no_server[0]; //total number of people in system is equal to initial number of agents

		num_arrivals.assign(class_no + 1, 0); //0th index is used for the sum
		num_in_queue.assign(class_no + 1, 0); //0th index is used for the sum
		num_abandons.assign(class_no + 1, 0); //0th index is used for the sum
		queue_integral.assign(class_no + 1, 0); //0th index is used for the sum
		service_integral.assign(class_no + 1, 0); //0th index is used for the sum
		system_integral.assign(class_no + 1, 0); //0th index is used for the sum
		holding_cost.assign(class_no, 0); 
		waiting_cost.assign(class_no, 0); 
		total_cost = 0;
		interval = 0;
		
		t_arrival = generate_interarrival(interval); //first event should be an arrival
		t_depart = MaxTime;
		t_abandon = MaxTime;
		
		int increment = 0;
		
		if (decision_freq == 300){
			increment = num_interval/204;
		} else if (decision_freq == 60){
			increment = num_interval/1020;
		} else if (decision_freq == 30){
			increment = num_interval/2040;
		} else if (decision_freq == 20){
			increment = num_interval/3060;
		} else if (decision_freq == 15){
			increment = num_interval/4080;
		}

		int end = num_interval - increment + 1;
		for (int i = 1; i <= end; i+= increment){
			int ind = (i - 1)/increment;
			std::string filename;
			filename = "z";
			filename += std::to_string(i);
			nn_zs[ind].load(neural_network_folder_name, filename);
		}
		
		//main part of the simulation
		while (sim_clock < T){

			t_event = std::min({t_arrival, t_depart, t_abandon}); //the current event

			for (int i = 0; i < class_no + 1; i++) {
				queue_integral[i] += num_in_queue[i]*(t_event-sim_clock);
				service_integral[i] += num_in_service[i]*(t_event-sim_clock);
				system_integral[i] += num_in_system[i]*(t_event-sim_clock);
			}
			
			for (int i = 0; i < class_no; ++i){
				holding_cost[i] += num_in_queue[i+1]*holding_cost_rate[i]*(t_event-sim_clock); //here time is in terms of hours so we don't have to divide the cost rate to adjust 
				waiting_cost[i] += num_in_queue[i+1]*cost_rate[i]*(t_event - sim_clock);
				total_cost += num_in_queue[i+1]*cost_rate[i]*(t_event-sim_clock);
			} 

			//advance the time
			sim_clock = t_event; //time unit is seconds
			pre_interval = interval;
			interval = std::min(int(sim_clock*12),203);
			post_interval = interval;

			//if the current event is an arrival
			if (t_event == t_arrival) {
				std::uniform_real_distribution<double> uniform(0.0, 1.0); //lookup seed 
            	double arrival_seed = uniform(generator);
            	auto low = std::lower_bound(arr_cdf[interval].begin(), arr_cdf[interval].end(), arrival_seed); //which class has arrived
				int arrival_ind = low - arr_cdf[interval].begin();
				//handle arrival event
            	handle_arrival_event(interval, arrival_ind, pre_interval, post_interval);
			
			} 
			//if the current event is a departure 
			else if (t_event == t_depart) {
				std::uniform_real_distribution<double> uniform(0.0, 1.0); //lookup seed 
            	double departure_seed = uniform(generator);
				std::vector<double> numerator(class_no, 0);
				std::vector<double> temp(class_no, 0);
				std::vector<double> ser_cdf;
				for (int i = 0; i < class_no; i++){
					numerator[i] = num_in_service[i+1] * mu_hourly[i]; 
				}
				double initial_sum = 0;
				double service_rate =  accumulate(numerator.begin(), numerator.end(), initial_sum);

				for (int i = 0; i < class_no; i++){
					temp[i] = numerator[i] / service_rate;
				}
				double s = 0;
				for (int i = 0; i < class_no; i++){
					s += temp[i];
					ser_cdf.push_back(s);
				}
				for (int i = 0; i < class_no; i++){
					double ind = ser_cdf[i];
				}
				auto low = std::lower_bound(ser_cdf.begin(), ser_cdf.end(), departure_seed); 
				int service_ind = low - ser_cdf.begin();
            	//handle service completion event - departure
				handle_depart_event(interval, service_ind, pre_interval, post_interval);				
			} 
			//if the current event is an abandonment
			else if (t_event == t_abandon) {
				handle_abandon_event(interval, pre_interval, post_interval);
			} 
			else {std::cout << "Something is Wrong" << std::endl;}
 
 		}

		//saving the ending cost per class and the total cost to res vector
		std::vector<double> res;
		res.assign(class_no + 1, 0);

		for (int i = 0; i < class_no; i ++){
			res[i] = waiting_cost[i] + overtime_cost * num_in_queue[i + 1]; //adding terminal cost to the waiting cost for each class
		}
		res[class_no] = total_cost + overtime_cost * num_in_queue[0]; //adding terminal cost to the total cost
       	return res;
	}
}

int main(int argc, char** argv){ 

	std::string jsonFileName = "/project/Call_Center_Control/analyses_final/nn_simulations/config.json"; //the configuration file to initialize the simulation
	std::string record_file = "/project/Call_Center_Control/analyses_final/nn_simulations/nn_policy.csv";  

	simulation::Simulation simObj(jsonFileName);
    simObj.save(record_file);	
	
	return 0;
}
