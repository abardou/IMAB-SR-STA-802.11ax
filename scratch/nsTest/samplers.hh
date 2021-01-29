#ifndef __SAMPLERS_HH
#define __SAMPLERS_HH

#include <vector>
#include <tuple>
#include <algorithm>
#include <random>
#include <chrono>

#include "parameters.hh"

using NetworkConfiguration = std::vector<std::tuple<double, double>>;

class Sampler {
	public:
		Sampler(const std::vector<ConstrainedCouple>& parameters);
		virtual ~Sampler();
		virtual NetworkConfiguration randomAction(const std::vector<NetworkConfiguration>& forbidden = std::vector<NetworkConfiguration>()) const;
		virtual void addToBase(NetworkConfiguration features, double reward);
		virtual NetworkConfiguration randomActionFromSample(const std::vector<NetworkConfiguration>& forbidden = std::vector<NetworkConfiguration>()) = 0;
		virtual NetworkConfiguration operator()(const std::vector<NetworkConfiguration>& forbidden = std::vector<NetworkConfiguration>()) = 0;

	public:
		std::vector<ConstrainedCouple> _parameters;
		std::vector<NetworkConfiguration> _features;
		std::vector<double> _rewards;
};

class RandomSampler : public Sampler {
	public:
		RandomSampler(const std::vector<ConstrainedCouple>&);

	protected:
		std::default_random_engine _generator;
		unsigned _seed = std::chrono::system_clock::now().time_since_epoch().count();
};

class UniformSampler : public RandomSampler {
	public:
		UniformSampler(const std::vector<ConstrainedCouple>& parameters);
		virtual NetworkConfiguration randomActionFromSample(const std::vector<NetworkConfiguration>& forbidden = std::vector<NetworkConfiguration>());
		virtual NetworkConfiguration operator()(const std::vector<NetworkConfiguration>& forbidden = std::vector<NetworkConfiguration>());
};


using Gaussian = std::tuple<std::vector<double>, unsigned int, double>;
using StandardDeviations = std::vector<double>;

using GaussianT = std::tuple<std::vector<double>, double, double>;

class HGMTSampler : public RandomSampler {
	public:
		HGMTSampler(const std::vector<ConstrainedCouple>& parameters, NetworkConfiguration& def, unsigned int nMax, double eps, double dist, unsigned int (*nTests)(std::vector<GaussianT>));
		void addGaussian(const NetworkConfiguration& center, double dist, double reward);
		void increaseStds();
		NetworkConfiguration rebuildConfiguration(const std::vector<double>& sampled) const;
		std::vector<double> normedSample(std::discrete_distribution<int>& discreteDist);
		virtual void addToBase(NetworkConfiguration features, double reward);
		virtual NetworkConfiguration randomActionFromSample(const std::vector<NetworkConfiguration>& forbidden = std::vector<NetworkConfiguration>());
		virtual NetworkConfiguration operator()(const std::vector<NetworkConfiguration>& forbidden = std::vector<NetworkConfiguration>());

	protected:
		unsigned int (*_nTests)(std::vector<GaussianT>);
		std::vector<GaussianT> _gaussians;
		double _eps;
		double _dist;
		unsigned int _nMax;
		unsigned int _testCounter;
};

#endif