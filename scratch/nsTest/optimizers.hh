#ifndef __OPTIMIZERS_HH
#define __OPTIMIZERS_HH

#include <vector>
#include <string>
#include <map>
#include <chrono>

#include "samplers.hh"

using History = std::vector<std::tuple<NetworkConfiguration, double>>;
class Optimizer {
	public:
		Optimizer(Sampler* sampler);
		virtual ~Optimizer();
		void showDecisions() const;
		virtual void addToBase(NetworkConfiguration configuration, double reward);
		virtual NetworkConfiguration optimize() = 0;

	protected:
		Sampler* _sampler;
		History _history;
		std::default_random_engine _generator;
		unsigned _seed = std::chrono::system_clock::now().time_since_epoch().count();
};

class EpsilonGreedyOptimizer : public Optimizer {
	public:
		EpsilonGreedyOptimizer(Sampler* sampler, double epsilon);
		virtual void addToBase(NetworkConfiguration configuration, double reward);
		virtual NetworkConfiguration optimize();

	protected:
		double _epsilon;
		std::map<NetworkConfiguration, double> _results;
		std::uniform_real_distribution<double> _distribution;
};

using GammaNormalSample = std::tuple<double, double, double, double, std::vector<double>, bool>;
class ThompsonGammaNormalOptimizer : public Optimizer {
	public:
		ThompsonGammaNormalOptimizer(Sampler* sampler, unsigned int sampleSize, double add);
		virtual void addToBase(NetworkConfiguration configuration, double reward);
		virtual NetworkConfiguration optimize();

	protected:
		std::map<NetworkConfiguration, GammaNormalSample> _gammaNormals;
		unsigned int _sampleSize;
		double _add;
};

class IdleOptimizer : public Optimizer {
	public:
		IdleOptimizer();
		virtual void addToBase(NetworkConfiguration configuration, double reward);
		virtual NetworkConfiguration optimize();

	protected:
		NetworkConfiguration _config;
};

class ThompsonGammaNormalWindowOptimizer : public Optimizer {
	public:
		ThompsonGammaNormalWindowOptimizer(Sampler* sampler, unsigned int sampleSize, unsigned int windowSize);
		virtual void addToBase(NetworkConfiguration configuration, double reward);
		virtual NetworkConfiguration optimize();

	protected:
		unsigned int _sampleSize;
		unsigned int _windowSize;
};

using NormalParameters = std::tuple<double, double, unsigned int>;
class ThompsonNormalOptimizer : public Optimizer {
	public:
		ThompsonNormalOptimizer(Sampler* sampler, double add);
		virtual void addToBase(NetworkConfiguration configuration, double reward);
		virtual NetworkConfiguration optimize();

	protected:
		std::map<NetworkConfiguration, NormalParameters> _normals;
		double _add;
};

using BetaParameters = std::tuple<double, double>;
class ThompsonBetaOptimizer : public Optimizer {
	public:
		ThompsonBetaOptimizer(Sampler* sampler, double add);
		virtual void addToBase(NetworkConfiguration configuration, double reward);
		virtual NetworkConfiguration optimize();

	protected:
		std::map<NetworkConfiguration, BetaParameters> _betas;
		double _add;
};

#endif