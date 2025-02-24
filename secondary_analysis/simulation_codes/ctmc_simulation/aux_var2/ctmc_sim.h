#ifndef CTMC_H
#define CTMC_H

#include <set>
#include <iostream>
#include <math.h>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <fstream>
#include <iterator>
#include <string>
#include <limits>
#include <list>
#include <vector>
#include <deque>
#include <stdlib.h>
#include <stdio.h>
#include <array>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#include <utility>
#include <cstdio>
#include <sstream>
#include <map>
#include <chrono>
#include <omp.h>
#include <random>
#include </var2_test_auxiliary_mdp/include/nlohmann/json.hpp>

namespace simulation
{
	class Simulation
	{
		public: 
			Simulation(const std::string& jsonFileName);
			~Simulation();

			int save();

			std::vector<std::string> splitString(const std::string& input, char delimiter);
			std::vector<std::vector<double>> readMatrixFromCSV(const std::string& filename); 
			std::vector<double> readVectorFromCSV(const std::string& filename);

		private:

			int class_no;
			int num_interval;
			int decision_freq;
			int num_iterations;
			int increment;
			std::vector<int> upper;

			std::vector<double> lambda;
			std::vector<int> no_server; 

			std::vector<double> mu_hourly;
			std::vector<double> theta_hourly;

			std::vector<double> holding_cost_rate;
			std::vector<double> abandonment_cost_rate;
			std::vector<double> cost_rate;
			std::vector<double> initialization;
			std::vector<std::vector<double>> arr_cdf; 
			std::vector<std::vector<float>> optimal_policy_files;
			std::vector<std::vector<std::string>> record;
	};

	class Execute
	{
		public: 
			Execute(int& class_no_,
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
							std::vector<std::string> index_);
			~Execute();

			std::vector<double> run(std::vector<double> initialization);
			
		private:

			void queue_init();
			double generate_interarrival(int& interval);
			double generate_abandon(int& cls);
			double generate_service();

			void add_queue(double& arr_time, int& cls);
			void remove_queue(int& cls);


			void handle_arrival_event(int& interval, int& cls, int& pre_interval, int& post_interval, std::vector<float>& optimal_policy_matrix);
			void handle_depart_event(int& interval, int& cls, int& pre_interval, int& post_interval, std::vector<float>& optimal_policy_matrix);
			void handle_abandon_event(int& interval, int& pre_interval, int& post_interval, std::vector<float>& optimal_policy_matrix);
			std::vector<double> argsort(const std::vector<double>&);
			void optimal_policy_calculation(std::vector<float>& optimal_policy_matrix);

			double t_arrival;
			double t_depart;
			double t_abandon;
			double t_event;
			double sim_clock = 0;
            double T = 17;
            double total_cost;

			int class_abandon;
			int cust_abandon;
			int interval;
			int pre_interval;
			int post_interval;
			int num_interval;
			int class_no;
			int decision_freq;
			int increment;

			std::vector<int> num_in_system;
			std::vector<int> num_in_service;
			std::vector<int> num_in_queue;
			std::vector<int> num_abandons;
			std::vector<int> num_arrivals;
			std::vector<std::string> index;
			std::vector<int> upper;

			std::vector<double> queue_integral;
			std::vector<double> service_integral;
			std::vector<double> system_integral;
			std::vector<double> holding_cost;
			std::vector<double> waiting_cost;

			std::vector<std::deque<double>> queue_list;
			std::vector<std::vector<double>> abandonment_list;

			std::deque<double> empty_queue;
			std::vector<std::deque<double> > arr_list;
			std::vector<float> optimal_policy_matrix;

			double inf = std::numeric_limits<double>::infinity();
			std::mt19937 generator;

			std::vector<double> mu_hourly;
			std::vector<double> theta_hourly;
			std::vector<double> lambda;
			std::vector<int> no_server; 
			std::vector<double> holding_cost_rate;
			std::vector<double> abandonment_cost_rate;
			std::vector<double> cost_rate;
			std::vector<std::vector<double>> arr_cdf; 
			std::vector<std::vector<float>> optimal_policy_files;
			
			
	};
};
#endif