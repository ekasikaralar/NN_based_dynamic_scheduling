#include "static.h"

namespace simulation {

	//initialization 
	Simulation::Simulation(const std::string& jsonFileName, std::string& rule){
		
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
    		class_no = config["class_no"];
    		num_interval = config["num_interval"];
		num_iterations = config["num_iterations"];
		priority_rule = rule;
    	
		std::string lambda_path = config["lambda_path"];
		std::string agents_path = config["agents_path"];
		std::string mu_hourly_path = config["mu_hourly_path"];
		std::string theta_hourly_path = config["theta_hourly_path"];
		std::string arr_cdf_path = config["arr_cdf_path"];
		std::string holding_cost_rate_path = config["holding_cost_rate_path"];
		std::string abandonment_cost_rate_path = config["abandonment_cost_rate_path"];
		std::string cost_rate_path = config["cost_rate_path"];
		std::string initialization_path = config["initialization_path"];

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
          									holding_cost_rate, abandonment_cost_rate, cost_rate, num_interval, iter, priority_rule); 

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
		return 0;
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
							int& num_interval_,
							int& seed, 
							std::string& priority_rule_)
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
		num_interval = num_interval_;
		priority_rule = priority_rule_;
        
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

	/**
 		* Determines the priority order for classes based on their cost rate.
 		*
 		* This function calculates a priority order by sorting the classes based on their cost rates 
 		*
 		* @param num_in_system The number of items in the system for each class 
 		* @param interval The current time interval 
 		* @return A vector of class indices sorted by their priority based on cost rate
 	*/
	std::vector<double> Execute::queueing_discipline(std::vector<int>& num_in_system, int& interval){

		std::vector<double> mu_theta_diff;
		std::vector<double> c_mu_theta_diff;
		std::vector<double> c_mu_theta;
		std::vector<double> c_mu;
		std::vector<double> kappa;

		mu_theta_diff.assign(class_no, 0);
		c_mu_theta_diff.assign(class_no, 0);
		c_mu_theta.assign(class_no, 0);
		c_mu.assign(class_no, 0);

		for (int i = 0; i < class_no; i++){
			c_mu_theta[i] = cost_rate[i]*mu_hourly[i]/theta_hourly[i]; //c*mu/theta rule
			c_mu[i] = cost_rate[i]*mu_hourly[i]; //c*mu rule
			c_mu_theta_diff[i] = cost_rate[i]*(mu_hourly[i] - theta_hourly[i]); //c*(mu - theta)rule
			mu_theta_diff[i] = mu_hourly[i] - theta_hourly[i]; //mu - theta rule
		}

		for (int i = 0; i < class_no; i++) {
			if (priority_rule == "cost"){
				kappa.push_back(-1 * (cost_rate[i])); //cost_rate
			} else if (priority_rule == "c_mu_theta"){
				kappa.push_back(-1 * (c_mu_theta[i])); //c*mu/theta rule
			} else if (priority_rule == "c_mu"){
				kappa.push_back(-1 * (c_mu[i])); //c*mu/theta rule
			} else if (priority_rule == "c_mu_theta_diff"){
				kappa.push_back(-1 * (c_mu_theta_diff[i])); //c*(mu - theta)rule
			} else if (priority_rule == "mu_theta_diff"){
				kappa.push_back(-1 * (mu_theta_diff[i])); //mu - theta rule
			}
		}

		std::vector<double> priority_order = argsort(kappa);
		return priority_order;
	}

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
		
		t_arrival = generate_interarrival(interval); //first event should be an arrival
		t_depart = MaxTime;
		t_abandon = MaxTime;
		
		//main part of the simulation
		while (sim_clock < T){

			t_event = std::min({t_arrival, t_depart, t_abandon}); //the current event

			
			for (int i = 0; i < class_no + 1; i++) {
				queue_integral[i] += num_in_queue[i]*(t_event-sim_clock);
				service_integral[i] += num_in_service[i]*(t_event-sim_clock);
				system_integral[i] += num_in_system[i]*(t_event-sim_clock);
			}
			
			for (int i = 0; i < class_no; ++i){
				holding_cost[i] += num_in_queue[i+1]*holding_cost_rate[i]*(t_event-sim_clock);
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
            	auto low = std::lower_bound(arr_cdf[interval].begin(), arr_cdf[interval].end(), arrival_seed); //determining which class has arrived
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
				
				//determining which class has departed
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
			res[i] = waiting_cost[i] + overtime_cost * num_in_queue[i + 1];//adding terminal cost to the waiting cost for each class
		}
		res[class_no] = total_cost + overtime_cost * num_in_queue[0]; //adding terminal cost to the total cost
       	return res;
	}
}

int main(int argc, char** argv){ 

	std::vector<std::string> rules = {
        "cost",
        "c_mu_theta",
        "c_mu_theta_diff",
        "c_mu",
        "mu_theta_diff"
    };


	std::string jsonFileName = "/project/Call_Center_Control/analyses_final/static_benchmark/config.json"; //the configuration file to initialize the simulation
	
	for (const auto& ruleConst : rules) {
		std::string rule = ruleConst;  
        std::string record_file = "/project/Call_Center_Control/analyses_final/static_benchmark/static_benchmark_" + rule + ".csv";  
        simulation::Simulation simObj(jsonFileName, rule);
        simObj.save(record_file);
        std::cout << "Simulation completed for rule: " << rule << std::endl;
    }

	return 0;
}