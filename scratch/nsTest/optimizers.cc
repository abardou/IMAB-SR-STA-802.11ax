#include "optimizers.hh"
#include <iostream>

/**
 * Build an Optimizer
 * 
 * @param sampler Sampler* the sampler to use to get new actions
 */
Optimizer::Optimizer(Sampler* sampler): _sampler(sampler), _generator(_seed) {  };

/**
 * Virtual destructor for Optimizer
 */
Optimizer::~Optimizer() {  }

/**
 * Add a configuration and its associated reward to the history.
 * 
 * @param configuration NetworkConfiguration the network configuration
 * @param reward double the reward associated
 */
void Optimizer::addToBase(NetworkConfiguration configuration, double reward) {
	this->_history.push_back(std::make_tuple(configuration, reward));
	if (this->_sampler != nullptr) {
		this->_sampler->addToBase(configuration, reward);
	}
}

/**
 * Show the average reward and the number of times each configuration is played
 */
void Optimizer::showDecisions() const {
	std::map<NetworkConfiguration, double> rewards;
	std::map<NetworkConfiguration, double> counters;
	std::vector<NetworkConfiguration> keys;
	// Compute statistics
	for (std::tuple<NetworkConfiguration, double> t: this->_history) {
		NetworkConfiguration conf = std::get<0>(t);
		double r = std::get<1>(t);

		if (rewards.find(conf) != rewards.end()) {
			rewards[conf] = (counters[conf] * rewards[conf] + r) / (counters[conf] + 1);
			counters[conf]++;
		} else {
			rewards[conf] = r;
			counters[conf] = 1;
			keys.push_back(conf);
		}
	}

	// Log the stats
	std::cout << "Rewards: [";
	for (NetworkConfiguration conf: keys) {
		std::cout << " " << (round(1000.0*rewards[conf])/1000.0) << ",";
	}
	std::cout << " ]" << std::endl << "Counter: [";
	for (NetworkConfiguration conf: keys) {
		std::cout << " " << counters[conf] << ",";
	}
	std::cout << " ]" << std::endl;
} 

IdleOptimizer::IdleOptimizer() : Optimizer(nullptr) {  }

void IdleOptimizer::addToBase(NetworkConfiguration configuration, double reward) {
	Optimizer::addToBase(configuration, reward);

	this->_config = configuration;
}

NetworkConfiguration IdleOptimizer::optimize() {
	return this->_config;
}

/**
 * Build an EpsilonGreedyOptimizer
 * 
 * @param sampler Sampler* the sampler to use to get new actions
 * @param epsilon double the exploration parameter
 */
EpsilonGreedyOptimizer::EpsilonGreedyOptimizer(Sampler* sampler, double epsilon): Optimizer(sampler), _epsilon(epsilon), _distribution(0.0, 1.0) {  };


/**
 * Add an observation to the optimizer and its associated sampler.
 * 
 * @param configuration NetworkConfiguration the network configuration
 * @param reward double the reward associated to the configuration 
 */
void EpsilonGreedyOptimizer::addToBase(NetworkConfiguration configuration, double reward) {
	Optimizer::addToBase(configuration, reward);

	if (this->_results.find(configuration) != this->_results.end())
		this->_results[configuration] = 0.8 * this->_results[configuration] + 0.2 * reward;
	else
		this->_results[configuration] = reward;
}

/**
 * Find the optimal configuration according to e-greedy strategy.
 * 
 * @return the optimal configuration according to e-greedy strategy
 */
NetworkConfiguration EpsilonGreedyOptimizer::optimize() {
	// this->showDecisions();

	// Explore or exploit
	bool explore = this->_distribution(this->_generator) < this->_epsilon;
	if (explore) {
		NetworkConfiguration sampled = (*this->_sampler)();
		if (std::get<0>(sampled[0]) != 0)
			return sampled;
	}

	// Exploitation
	double max = 0;
	NetworkConfiguration confMax;
	for (std::map<NetworkConfiguration, double>::iterator it = this->_results.begin(); it != this->_results.end(); ++it) {
		double challenger = it->second;
		if (max < challenger) {
			max = challenger;
			confMax = it->first;
		}
	}

	return confMax;
}

/**
 * Build a ThompsonGammaNormalOptimizer
 * 
 * @param sampler Sampler* the sampler to draw new configurations from
 * @param sampleSize unsigned int the sample size to use for update
 * @param add double the exploration parameter
 */
ThompsonGammaNormalOptimizer::ThompsonGammaNormalOptimizer(Sampler* sampler, unsigned int sampleSize, double add): Optimizer(sampler), _sampleSize(sampleSize), _add(add) {  }

/**
 * Add an observation to the optimizer and its associated sampler.
 * 
 * @param configuration NetworkConfiguration the network configuration
 * @param reward double the reward associated to the configuration 
 */
void ThompsonGammaNormalOptimizer::addToBase(NetworkConfiguration configuration, double reward) {
	Optimizer::addToBase(configuration, reward);

	// Search in attribute for a preexisting sample
	if (this->_gammaNormals.find(configuration) != this->_gammaNormals.end()) {
		GammaNormalSample& lns = this->_gammaNormals[configuration]; 
		std::vector<double>& sample = std::get<4>(lns);
		sample.push_back(reward);
		// Update the gamma normal if we reach sample size
		if (sample.size() == this->_sampleSize) {
			double mean = std::accumulate(sample.begin(), sample.end(), 0.0) / this->_sampleSize;
			double var = this->_sampleSize * (std::inner_product(sample.begin(), sample.end(), sample.begin(), 0.0) / this->_sampleSize - mean * mean) / (this->_sampleSize - 1);
			double &mu = std::get<0>(lns),
						 &lambda = std::get<1>(lns),
						 &alpha = std::get<2>(lns),
						 &beta = std::get<3>(lns),
						 newMu, newLambda, newAlpha, newBeta;
			bool& explore = std::get<5>(lns);
			if (explore) {
				newMu = mean;
				newLambda = this->_sampleSize;
				newAlpha = this->_sampleSize / 2.0;
				newBeta = this->_sampleSize * var / 2.0;
				explore = false;
			} else {
				newMu = (lambda * mu + this->_sampleSize * mean) / (lambda + this->_sampleSize);
				newLambda = lambda + this->_sampleSize;
				newAlpha = alpha + this->_sampleSize / 2.0;
				newBeta = beta + (this->_sampleSize * var + lambda * this->_sampleSize * pow(mean - mu, 2) / (lambda + this->_sampleSize)) / 2.0;
			}

			mu = newMu;
			lambda = newLambda;
			alpha = newAlpha;
			beta = newBeta;
			sample.clear();
		}
 	} else {
		 // Create a new instance in logNormals
		this->_gammaNormals[configuration] = std::make_tuple(0.5, 1.0, 0.5, 0.025, std::vector<double>({reward}), false);
	}
}

/**
 * Find the best configuration according to ThompsonGammaNormal strategy
 * 
 * @return the best configuration according to ThompsonGammaNormal strategy 
 */
NetworkConfiguration ThompsonGammaNormalOptimizer::optimize() {
	// this->showDecisions();

	// Explore or exploit
	bool explore = this->_gammaNormals.size() < pow(this->_history.size(), this->_add / (this->_add + 1));
	if (explore) {
		// Request a new configuration to the sampler
		std::vector<NetworkConfiguration> forbidden;
		for (std::map<NetworkConfiguration, GammaNormalSample>::iterator it = this->_gammaNormals.begin(); it != this->_gammaNormals.end(); ++it)
			forbidden.push_back(it->first);
		NetworkConfiguration sampled = (*this->_sampler)(forbidden);
		if (std::get<0>(sampled[0]) != 0)
			return sampled;
	}

	// Look for not enough explored configurations
	std::vector<NetworkConfiguration> toExplore;
	for (std::map<NetworkConfiguration, GammaNormalSample>::iterator it = this->_gammaNormals.begin(); it != this->_gammaNormals.end(); ++it)
		if (std::get<5>(it->second))
			toExplore.push_back(it->first);
	if (!toExplore.empty()) {
		// Test a not enough explored configuration
		std::uniform_int_distribution<> d(0, toExplore.size()-1);
		return toExplore[d(this->_generator)];
	} else {
		// Sample taus for normal distributions
		std::vector<double> mus(this->_gammaNormals.size());
		int i = 0;
		for (std::map<NetworkConfiguration, GammaNormalSample>::iterator it = this->_gammaNormals.begin(); it != this->_gammaNormals.end(); ++it) {
			std::gamma_distribution<> gamma(std::get<2>(it->second), 1.0 / std::get<3>(it->second));
			double tau = gamma(this->_generator);
			std::normal_distribution<> normal(std::get<0>(it->second), 1.0 / sqrt(std::get<1>(it->second) * tau));
			mus[i] = normal(this->_generator);
			i++;
		}
		int maxElementIndex = std::max_element(mus.begin(), mus.end()) - mus.begin();
		std::map<NetworkConfiguration, GammaNormalSample>::iterator maxConf = this->_gammaNormals.begin();
		for (int i = 0; i < maxElementIndex; i++) maxConf++;

		return maxConf->first;
	}
}

/**
 * Build a ThompsonGammaNormalWindowOptimizer
 * 
 * @param sampler Sampler* the sampler to draw new configurations from
 * @param sampleSize unsigned int the sample size to use for update
 * @param add double the exploration parameter
 */
ThompsonGammaNormalWindowOptimizer::ThompsonGammaNormalWindowOptimizer(Sampler* sampler, unsigned int sampleSize, unsigned int windowSize): Optimizer(sampler), _sampleSize(sampleSize), _windowSize(windowSize) {  }

/**
 * Add an observation to the optimizer and its associated sampler.
 * 
 * @param configuration NetworkConfiguration the network configuration
 * @param reward double the reward associated to the configuration 
 */
void ThompsonGammaNormalWindowOptimizer::addToBase(NetworkConfiguration configuration, double reward) {
	Optimizer::addToBase(configuration, reward);
}

/**
 * Find the best configuration according to ThompsonGammaNormal strategy
 * 
 * @return the best configuration according to ThompsonGammaNormal strategy 
 */
NetworkConfiguration ThompsonGammaNormalWindowOptimizer::optimize() {
	// this->showDecisions();

	// Get the window
	unsigned int historySize = std::min(this->_windowSize, (unsigned int) this->_history.size());
	History window = History(this->_history.end() - historySize, this->_history.end());

	// std::cout << "WINDOW: ";
	// for (std::tuple<NetworkConfiguration, double> t: window) {
	// 	std::cout << std::get<1>(t) << " ";
	// }
	// std::cout << std::endl;
	// Get the candidates
	std::vector<NetworkConfiguration> uniques;
	std::vector<std::tuple<unsigned int, double, double>> stats;
	for (std::tuple<NetworkConfiguration, double> t: window) {
		NetworkConfiguration c = std::get<0>(t);
		double rew = std::get<1>(t);
		std::vector<NetworkConfiguration>::iterator it = std::find(uniques.begin(), uniques.end(), c);
		if (it == uniques.end()) {
			uniques.push_back(c);
			stats.push_back(std::make_tuple(1, rew, rew * rew));
		} else {
			unsigned int idx = std::distance(uniques.begin(), it);
			std::get<0>(stats[idx]) += 1;
			std::get<1>(stats[idx]) += rew;
			std::get<2>(stats[idx]) += rew * rew;			
		}
	}

	// Explore or exploit
	std::uniform_real_distribution<> real_dist;
	bool explore = real_dist(this->_generator) < 0.1;
	if (explore) {
		// Request a new configuration to the sampler
		std::vector<NetworkConfiguration> forbidden;
		for (NetworkConfiguration c: uniques)
			forbidden.push_back(c);
		NetworkConfiguration sampled = (*this->_sampler)(forbidden);
		if (std::get<0>(sampled[0]) != 0)
			return sampled;
	}

	// Look for not enough explored configurations
	std::vector<NetworkConfiguration> toExplore;
	for (unsigned int i = 0; i < uniques.size(); i++)
		if (std::get<0>(stats[i]) < this->_sampleSize) {
			unsigned int idxAppearance = 0;
			for (std::tuple<NetworkConfiguration, double> t: this->_history) {
				if (std::get<0>(t) == uniques[i]) {
					break;
				}
				idxAppearance++;
			}
			if (historySize < this->_windowSize || idxAppearance / this->_windowSize > 0.5)
				toExplore.push_back(uniques[i]);
		}
	if (!toExplore.empty()) {
		// Test a not enough explored configuration
		std::uniform_int_distribution<> d(0, toExplore.size()-1);
		return toExplore[d(this->_generator)];
	} else {
		// Sample taus for normal distributions
		std::vector<double> mus(uniques.size());
		int i = -1;
		for (NetworkConfiguration c: uniques) {
			i++;
			// Compute NormalGamma parameters
			unsigned int size = std::get<0>(stats[i]);
			if (size == 1) {
				mus[i] = -1000;
				continue;
			}
			double mean = std::get<1>(stats[i]) / size, var = std::get<2>(stats[i]) / size - mean * mean;
			double mu = mean,
						 lambda = size,
						 alpha = size / 2.0,
						 beta = size * var / 2.0;
			std::gamma_distribution<> gamma(alpha, 1.0 / beta);
			double tau = gamma(this->_generator);
			std::normal_distribution<> normal(mu, 1.0 / sqrt(lambda * tau));
			mus[i] = normal(this->_generator);
		}
		int maxElementIndex = std::max_element(mus.begin(), mus.end()) - mus.begin();

		return uniques[maxElementIndex];
	}
}


/**
 * Build a ThompsonNormalOptimizer
 * 
 * @param sampler Sampler* the sampler to draw new configurations from
 * @param add double the exploration parameter
 */
ThompsonNormalOptimizer::ThompsonNormalOptimizer(Sampler* sampler, double add): Optimizer(sampler), _add(add) {  }

/**
 * Add an observation to the optimizer and its associated sampler.
 * 
 * @param configuration NetworkConfiguration the network configuration
 * @param reward double the reward associated to the configuration 
 */
void ThompsonNormalOptimizer::addToBase(NetworkConfiguration configuration, double reward) {
	Optimizer::addToBase(configuration, reward);

	// Search in attribute for a preexisting sample
	if (this->_normals.find(configuration) != this->_normals.end()) {
		NormalParameters& nps = this->_normals[configuration];
		double &mean = std::get<0>(nps),
					 &var = std::get<1>(nps);
		unsigned int &n = std::get<2>(nps);
		// Update the normal
		mean = (n * mean + reward) / (n + 1);
		var = 1.0 / (n + 1);
		n = n + 1;
 	} else {
		 // Create a new instance in normals
		this->_normals[configuration] = std::make_tuple(0, 1, 1);
	}
}

/**
 * Find the best configuration according to ThompsonNormal strategy
 * 
 * @return the best configuration according to ThompsonNormal strategy 
 */
NetworkConfiguration ThompsonNormalOptimizer::optimize() {
	// this->showDecisions();

	// Explore or exploit
	bool explore = this->_normals.size() < pow(this->_history.size(), this->_add / (this->_add + 1));
	if (explore) {
		// Request a new configuration to the sampler
		std::vector<NetworkConfiguration> forbidden;
		for (std::map<NetworkConfiguration, NormalParameters>::iterator it = this->_normals.begin(); it != this->_normals.end(); ++it)
			forbidden.push_back(it->first);
		NetworkConfiguration sampled = (*this->_sampler)(forbidden);
		if (std::get<0>(sampled[0]) != 0)
			return sampled;
	}

	// Sample mus for normal distributions
	std::vector<double> mus(this->_normals.size());
	int i = 0;
	for (std::map<NetworkConfiguration, NormalParameters>::iterator it = this->_normals.begin(); it != this->_normals.end(); ++it) {
		std::normal_distribution<> normal(std::get<0>(it->second), sqrt(std::get<1>(it->second)));
		mus[i] = normal(this->_generator);
		i++;
	}
	int maxElementIndex = std::max_element(mus.begin(), mus.end()) - mus.begin();
	std::map<NetworkConfiguration, NormalParameters>::iterator maxConf = this->_normals.begin();
	for (int i = 0; i < maxElementIndex; i++) maxConf++;

	return maxConf->first;
}

/**
 * Build a ThompsonBetaOptimizer
 * 
 * @param sampler Sampler* the sampler used to get new configurations
 * @param add double handle the exploration of the agent
 */
ThompsonBetaOptimizer::ThompsonBetaOptimizer(Sampler* sampler, double add) : Optimizer(sampler), _add(add) {  }

/**
 * Add a new configuration to the base, modifying the corresponding beta distribution.
 * 
 * @param configuration NetworkConfiguration the network configuration to add
 * @param reward double the reward associated to the configuration 
 */
void ThompsonBetaOptimizer::addToBase(NetworkConfiguration configuration, double reward) {
	Optimizer::addToBase(configuration, reward);
	// Bernoulli experiment to modify the configuration
	std::bernoulli_distribution bernDist(reward);
	bool bern = bernDist(this->_generator);
	int da = bern, db = 1 - bern;
	// Does the configuration already exist?
	if (this->_betas.find(configuration) != this->_betas.end()) {
		std::get<0>(this->_betas[configuration]) += da;
		std::get<1>(this->_betas[configuration]) += db;
	} else
		this->_betas[configuration] = std::make_tuple(1.0+da, 1.0+db);
}

/**
 * Find the best configuration to test according to the ThompsonBeta strategy
 * 
 * @return the best configuration to test according to ThompsonBeta
 */
NetworkConfiguration ThompsonBetaOptimizer::optimize() {
	// this->showDecisions();

	// Exploration or exploitation?
	bool explore = this->_betas.size() < pow(this->_history.size(), this->_add / (this->_add + 1));
	if (explore) {
		// Request a new configuration to the sampler
		std::vector<NetworkConfiguration> forbidden;
		for (std::map<NetworkConfiguration, BetaParameters>::iterator it = this->_betas.begin(); it != this->_betas.end(); ++it)
			forbidden.push_back(it->first);
		NetworkConfiguration sampled = (*this->_sampler)(forbidden);
		if (std::get<0>(sampled[0]) != 0)
			return sampled;
	}

	std::vector<double> scores(this->_betas.size());
	int i = 0;
	for (std::map<NetworkConfiguration, BetaParameters>::iterator it = this->_betas.begin(); it != this->_betas.end(); ++it) {
		std::gamma_distribution<> gammaX(std::get<0>(it->second), 1);
		std::gamma_distribution<> gammaY(std::get<1>(it->second), 1);
		double X = gammaX(this->_generator), Y = gammaY(this->_generator), Z = X / (X + Y);
		scores[i] = Z;
		i++;
	}
	int maxElementIndex = std::max_element(scores.begin(), scores.end()) - scores.begin();
	std::map<NetworkConfiguration, BetaParameters>::iterator maxConf = this->_betas.begin();
	for (int i = 0; i < maxElementIndex; i++) maxConf++;

	return maxConf->first;
}