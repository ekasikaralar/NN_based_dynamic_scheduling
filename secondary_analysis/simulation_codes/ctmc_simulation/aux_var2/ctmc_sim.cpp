#include "ctmc_sim.h"

namespace simulation {

	Simulation::Simulation(const std::string& jsonFileName){

		class_no = 17; //this simulation is designed for the 3 dimensional problems

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
		num_interval = config["num_interval"];
		num_iterations = config["num_iterations"]; //number of simulation replications
		decision_freq = config["decision_freq"]; //in seconds
		
		std::string lambda_path = config["lambda_path"];
		std::string agents_path = config["agents_path"];
		std::string mu_hourly_path = config["mu_hourly_path"];
		std::string theta_hourly_path = config["theta_hourly_path"];
		std::string arr_cdf_path = config["arr_cdf_path"];
		std::string holding_cost_rate_path = config["holding_cost_rate_path"];
		std::string abandonment_cost_rate_path = config["abandonment_cost_rate_path"];
		std::string cost_rate_path = config["cost_rate_path"];
		std::string initialization_path = config["initialization_path"];
		std::string upper_bound_path = config["upper_bound_path"];
		std::string policy_path = config["policy_path"]; //where the policy files are stored

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
		initialization = readVectorFromCSV(initialization_path);  //initialization of X(0)
		cost_rate = readVectorFromCSV(cost_rate_path); //hourly cost rate
		std::vector<double> upper_bound_vec = readVectorFromCSV(upper_bound_path); //upper bound of the state space
		upper.resize(upper_bound_vec.size());
		std::transform(upper_bound_vec.begin(), upper_bound_vec.end(), upper.begin(), 
                   [](double val) { return static_cast<int>(val); });
		unsigned long state_space_dim = upper[0] * upper[1] * upper[2]; //the dimensions of the state space
		optimal_policy_files = std::vector<std::vector<float>>(num_interval, std::vector<float>(state_space_dim, 0));

		if (decision_freq == 300){
			increment = num_interval/204;
		} else if (decision_freq == 60){
			increment = num_interval/1020;
		} else if (decision_freq == 30){
			increment = num_interval/2040;
		} else if (decision_freq == 15){
			increment = num_interval/4080;
		}
		
		int start = increment;
		
		for (int i = start; i <= num_interval; i+= increment){
			std::string policy_file;
			policy_file = policy_path + "policy_ln" + std::to_string(i) + ".txt";
			int ind = (i - increment)/increment;
			std::cout << "ind " << ind << std::endl;
			std::vector<double> doubleVec = readVectorFromCSV(policy_file);
			std::vector<float> floatVec(doubleVec.begin(), doubleVec.end());	
			optimal_policy_files[ind] = floatVec;
		}

		std::vector<std::string> policies = {"c_mu", "cost", "c_mu_theta", "c_mu_theta_diff", "mu_theta_diff"};

    	// Generate combinations
    	for (const auto& first : policies) {
        	for (const auto& second : policies) {
            	for (const auto& third : policies) {
					for (const auto& fourth : policies){
                		record.push_back({first, second, third, fourth});
                		if (record.size() == 625) { // Stop once we have 625 combinations
                    		break;
                		}
					}
					if (record.size() == 625) break;
            	}
            	if (record.size() == 625) break;
        	}
        	if (record.size() == 625) break;
		}
		
		// Output the matrix to check
    	for (const auto& row : record) {
        	for (const auto& element : row) {
            	std::cout << element << " ";
        	}
        	std::cout << std::endl;
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

	int Simulation::save(){
        
		for (int k = 0; k < 625; k++){

			std::string file_name_pol = "/var2_test_auxiliary_mdp/policy" + std::to_string(k) + ".csv";

			const char *path_pol = &file_name_pol[0];
			const int myfile_pol = open(path_pol, O_CREAT | O_WRONLY);

			if (myfile_pol != -1){			

				#pragma omp parallel for num_threads(800) // Adjust the number of threads as needed
				for (int iter = 0; iter < 10000; iter++){
					std::vector<double> cost; //to save the costs
					cost.assign(class_no + 1, 0);
					std::cout << "Iteration: " << iter << std::endl;

					// Execute simulation
					simulation::Execute exec(class_no, arr_cdf, lambda, mu_hourly, theta_hourly, no_server, optimal_policy_files,
          										holding_cost_rate, abandonment_cost_rate, cost_rate, num_interval, decision_freq, increment, upper, iter, record[k]); 

					cost = exec.run(initialization);

					std::string results;
					results += std::to_string(iter);
					results += ",";

					for (int i = 0; i < class_no; i++){
						results += std::to_string(cost[i]);
						results += ",";
					}

					results += std::to_string(cost[class_no]); //total_cost
					results += "\n";

					const char *char_results = const_cast<char*>(results.c_str());
					write(myfile_pol, char_results, results.length());
				}
			}

			close(myfile_pol);
		}
		return 0;
	}

	Execute::Execute(int& class_no_,
							std::vector<std::vector<double>>& arr_cdf_,
							std::vector<double>& lambda_,
							std::vector<double>& mu_hourly_,
							std::vector<double>& theta_hourly_, 
							std::vector<int>& no_server_,
							std::vector<std::vector<float>>& optimal_policy_files_,
							std::vector<double>& holding_cost_rate_,
							std::vector<double>& abandonment_cost_rate_,
							std::vector<double>& cost_rate_,
							int& num_interval_,
							int& decision_freq_,
							int& increment_,
							std::vector<int>& upper_,
							int& seed,
							std::vector<std::string> index_)
	{
		class_no = class_no_;

		arr_cdf = arr_cdf_; 
		lambda = lambda_;
		no_server = no_server_; 
		optimal_policy_files = optimal_policy_files_;

		mu_hourly = mu_hourly_; 
		theta_hourly = theta_hourly_;

		holding_cost_rate = holding_cost_rate_;
		abandonment_cost_rate = abandonment_cost_rate_;
		cost_rate = cost_rate_;
		num_interval = num_interval_;
		upper = upper_;
        decision_freq = decision_freq_;
		increment = increment_;
		index = index_;

      	generator.seed(seed);
		queue_init();

	}

	Execute::~Execute()
	{
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

	void Execute::optimal_policy_calculation(std::vector<float>& optimal_policy_matrix){
			
			std::vector<int> optimal_policy(class_no, 0);

			std::vector<double> mu_theta_diff;
			std::vector<double> c_mu_theta_diff;
			std::vector<double> c_mu_theta;
			std::vector<double> c_mu;
			std::vector<int> high_priority_group;
			std::vector<int> low_priority_group1;
			std::vector<int> low_priority_group2;
			std::vector<int> low_priority_group3;

			// high priority group 
			if (index[0] == "c_mu"){
				high_priority_group = {5, 4, 1, 0, 2, 6};
			} else if (index[0] == "cost"){
				high_priority_group = {5, 4, 1, 0, 2, 6};
			} else if (index[0] == "c_mu_theta"){
				high_priority_group = {5, 2, 0, 4, 6, 1};
			} else if (index[0] == "c_mu_theta_diff"){
				high_priority_group = {5, 2, 4, 0, 1, 6};
			} else if (index[0] == "mu_theta_diff"){
				high_priority_group = {2, 0, 6, 5, 1, 4};
			}

			// low priority group 1 
			if (index[3] == "c_mu"){
				low_priority_group3 = {12, 14, 15, 16};
			} else if (index[3] == "cost"){
				low_priority_group3 = {12, 14, 15, 16};
			} else if (index[3] == "c_mu_theta"){
				low_priority_group3 = {14, 16, 12, 15};
			} else if (index[3] == "c_mu_theta_diff"){
				low_priority_group3 = {14, 16, 15, 12};
			} else if (index[3] == "mu_theta_diff"){
				low_priority_group3 = {14, 16, 15, 12};
			}

			// low priority group 2
			if (index[1] == "c_mu"){
				low_priority_group1 = {3, 13, 10, 7};
			} else if (index[1] == "cost"){
				low_priority_group1 = {3, 13, 7, 10};
			} else if (index[1] == "c_mu_theta"){
				low_priority_group1 = {3, 7, 13, 10};
			} else if (index[1] == "c_mu_theta_diff"){
				low_priority_group1 = {7, 10, 13, 3};
			} else if (index[1] == "mu_theta_diff"){
				low_priority_group1 = {10, 7, 13, 3};
			}

			// low priority group 3
			if (index[2] == "c_mu"){
				low_priority_group2 = {9, 11, 8};
			} else if (index[2] == "cost"){
				low_priority_group2 = {9, 11, 8};
			} else if (index[2] == "c_mu_theta"){
				low_priority_group2 = {9, 8, 11};
			} else if (index[2] == "c_mu_theta_diff"){
				low_priority_group2 = {8, 11, 9};
			} else if (index[2] == "mu_theta_diff"){
				low_priority_group2 = {8, 9, 11};
			}

			std::vector<int> priority_order;
			
			//x1 number of people in low priority group 1
			//x2 number of people in low priority group 2
			//x3 number of people in low priority group 3
			int x1 = 0, x2 = 0, x3 = 0;

	
			// find the state
			// Summing up x1 for the low_priority_group1 elements
			for (int ind1 : low_priority_group1) {
    			x1 += num_in_system[ind1 + 1];
			}
			for (int ind2 : low_priority_group2) {
    			x2 += num_in_system[ind2 + 1];
			}

			for (int ind3 : low_priority_group3) {
    			x3 += num_in_system[ind3 + 1];
			}

			int opt_policy = 0;

			//calculating what state we are in -- in a sub2ind fashion
			unsigned long state = x1 + upper[0] * x2 + upper[0] * upper[1] * x3;
			
			if ((x1 > upper[0] - 1) || (x2 > upper[1] - 1) || (x3 > upper[2] - 1)){
				opt_policy = 1; //the default policy is serve group1, then group2 ,then group3
			} else {
			   opt_policy = optimal_policy_matrix[state];
			}

			// Define the possible orders of priority groups
			std::vector<std::vector<int>> priority_orders = {
    			{0, 1, 2}, // opt_policy == 1
    			{0, 2, 1}, // opt_policy == 2
    			{1, 0, 2}, // opt_policy == 3
    			{1, 2, 0}, // opt_policy == 4
    			{2, 0, 1}, // opt_policy == 5
    			{2, 1, 0}  // opt_policy == 6
			};

			// Combine all groups based on priority
			std::vector<std::vector<int>> low_priority_groups = {low_priority_group1, low_priority_group2, low_priority_group3};
			std::vector<int> all_priority_groups = high_priority_group;
			
			//in our matlab code, we use the following numbers to determine which policy is optimal:
			//if the optimal policy is 1: 1) group 1; 2) group 2; 3) group 3
			//if the optimal policy is 2: 1) group 1; 2) group 3; 3) group 2
			//if the optimal policy is 3: 1) group 2; 2) group 1; 3) group 3
			//if the optimal policy is 4: 1) group 2; 2) group 3; 3) group 1
			//if the optimal policy is 5: 1) group 3; 2) group 1; 3) group 2
			//if the optimal policy is 6: 1) group 3; 2) group 2; 3) group 1
			
			for (int idx : priority_orders[opt_policy - 1]) {
    			all_priority_groups.insert(all_priority_groups.end(), low_priority_groups[idx].begin(), low_priority_groups[idx].end());
			}

			// Set priority order directly
			priority_order = all_priority_groups;

			int avail_server_num;
			int num_served = 0;
			int ind;
		
			//we determine the optimal policy based on preemptive scheduling
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

			// Calculate the difference between current queue length and the optimal queue length
			for (int i = 0; i < class_no; i++){
				diff[i] = num_in_queue[i+1] - optimal_policy[i]; // difference -- if positive, people should not be placed in the queue and should be send to service; if negative, more people should be placed in the queue.
				num_in_service[i + 1] += diff[i]; // 
				num_in_queue[0] -= diff[i];
				num_in_queue[i + 1] -= diff[i];

				if (diff[i] >= 0){
					// Admit customers to service from class i 
					for (int j = 0; j < diff[i]; j++){
						remove_queue(i); // remove them from their corresponding queue
					}
				} else if (diff[i] < 0){
					// Add customers back to the queue
					for (int j = 0; j < abs(diff[i]); j++){
						add_queue(t_event,i);
					}
				}
			}

     	 	// Update total number of people in service
     	 	num_in_service[0] = 0;
     	 	for (int i = 0; i < class_no; i++){
     	 		num_in_service[0] += num_in_service[i + 1];
     	 	}
			
			// Schedule the next departure
			if (num_in_service[0] == 0) {
    			t_depart = std::numeric_limits<double>::infinity();
			} else {
    			t_depart = sim_clock + generate_service();
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
		* @param optimal_policy_matrix The matrix that contains the optimal policy for a given state
 	*/
	void Execute::handle_arrival_event(int& interval, int& cls, int& pre_interval, int& post_interval, std::vector<float>& optimal_policy_matrix){ 
		
		//an additional person is added to the system
	    num_in_system[0] += 1; 
	    num_in_system[cls + 1] += 1; 

		//arrival time for the next arrival event is scheduled
		t_arrival = sim_clock + generate_interarrival(interval);

		//an additional person is added to the queue first
		num_in_queue[0] += 1; 
	    num_in_queue[cls+1] += 1;
		num_arrivals[cls+1] += 1;
		num_arrivals[0] += 1;

		//an additional person is added to the queue_list
	    add_queue(t_event,cls);
		
		//if there are enough servers to serve everyone in the system, then no need to calculate the optimal policy

	    if ((num_in_system[0] <= no_server[interval] and pre_interval == post_interval) or ((num_in_service[0] < no_server[interval]) and pre_interval == post_interval)) {
		   	
			if (num_in_service[0] < no_server[interval]) {
			    //without waiting service entrance
				//someone is added to service
			    num_in_service[0] += 1;
			    num_in_service[cls + 1] += 1;

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
			optimal_policy_calculation(optimal_policy_matrix);			
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
		* @param optimal_policy_matrix The matrix that contains the optimal policy for a given state.
 	*/
	void Execute::handle_depart_event(int& interval, int& cls, int& pre_interval, int& post_interval, std::vector<float>& optimal_policy_matrix){ 
		
		num_in_system[0] -= 1; 
		num_in_system[cls+1] -=1; 	
		num_in_service[0] -= 1;
		num_in_service[cls+1] -= 1;
        
		optimal_policy_calculation(optimal_policy_matrix);
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
		* @param optimal_policy_matrix The matrix that contains the optimal policy for a given state.
 	*/
	void Execute::handle_abandon_event(int& interval, int& pre_interval, int& post_interval, std::vector<float>& optimal_policy_matrix){
	
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
           optimal_policy_calculation(optimal_policy_matrix);
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

	
	std::vector<double> Execute::run(std::vector<double> initialization)
	{	
    	constexpr double MaxTime = std::numeric_limits<double>::max();
		double overtime_cost = 2.12;

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
		int policy_interval = 0;
		
		t_arrival = generate_interarrival(interval); //first event should be an arrival
		t_depart = MaxTime;
		t_abandon = MaxTime;

		optimal_policy_matrix.assign(upper[0] * upper[1] * upper[2], 0);
		optimal_policy_matrix = optimal_policy_files[0];
		
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
			pre_interval = policy_interval;
			if (decision_freq == 300){
				policy_interval = std::min(int(sim_clock*12),203);
			} else if (decision_freq == 60){
				policy_interval = std::min(int(sim_clock*60),1019);
			} else if (decision_freq == 30){
				policy_interval = std::min(int(sim_clock*120),2039);
			} else if (decision_freq == 15){
				policy_interval = std::min(int(sim_clock*240),4079);
			}
			interval = std::min(int(sim_clock*12),203);
			post_interval = policy_interval;

			//read the new policy file
			if (pre_interval != post_interval){
				optimal_policy_matrix = optimal_policy_files[policy_interval];
			}
			//if the current event is an arrival
			if (t_event == t_arrival) {
				//std::cout << " arrival " << std::endl;
				std::uniform_real_distribution<double> uniform(0.0, 1.0); //lookup seed 
            	double arrival_seed = uniform(generator);
            	auto low = std::lower_bound(arr_cdf[interval].begin(), arr_cdf[interval].end(), arrival_seed); //which class has arrived
				int arrival_ind = low - arr_cdf[interval].begin();//handle arrival event
            	handle_arrival_event(interval, arrival_ind, pre_interval, post_interval, optimal_policy_matrix);
			
			}
			//if the current event is a departure 
			else if (t_event == t_depart) {
				//std::cout << " depart " << std::endl;
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
				auto low = std::lower_bound(ser_cdf.begin(), ser_cdf.end(), departure_seed); //which class has arrived
				int service_ind = low - ser_cdf.begin();//handle arrival event
            	//handle service completion event - departure
				handle_depart_event(interval, service_ind, pre_interval, post_interval, optimal_policy_matrix);				
			}
			//if the current event is an abandonment
			else if (t_event == t_abandon) {
				//std::cout << " abandon " << std::endl;
				handle_abandon_event(interval, pre_interval, post_interval, optimal_policy_matrix);
			} 
			else {std::cout << "Something is Wrong" << std::endl;}
		   	
		    //std::cout << "num_in_system: " <<  num_in_system[0] << " | num_in_queue: " <<  num_in_queue[0] << " | num_in_service: " <<  num_in_service[0] <<std::endl;
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

	std::string jsonFileName = "/var2_test_auxiliary_mdp/config_var2.json"; //the configuration file to initialize the simulation

	simulation::Simulation simObj(jsonFileName);
    simObj.save();	

	return 0;
}
