#ifndef __SIMULATION_HH
#define __SIMULATION_HH

#include <string>
#include <vector>
#include <set>
#include "unistd.h"

#include "json.hh"

#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/mobility-module.h"
#include "ns3/config-store-module.h"
#include "ns3/wifi-module.h"
#include "ns3/internet-module.h"
#include "ns3/applications-module.h"

#include "optimizers.hh"
#include "samplers.hh"

using namespace ns3;

enum Optim { EGREEDY, THOMP_BETA, THOMP_GAMNORM, THOMP_NORM, THOMP_GAMNORM_WINDOW, IDLEOPT };
enum Samp { UNIF, HGM };
enum Reward { AD_HOC, FSCORE };
enum Dist { LOG, SQRT, N2, N4 };

class Simulation {
	public:
		Simulation(Optim oId, Samp sId, Reward r, std::string topoPath, double duration, double testDuration, std::string outputName, std::vector<double> programSteps = {0.0}, std::vector<double> saturationProgram = {1.0}, unsigned int windowSize = 1, NetworkConfiguration defaultConf = {}, double beta=1.0);
		pid_t getPID() const;
		void readTopology(std::string path);
		void storeMetrics();
		double rewardFromThroughputs();
		double adHocRewardFromThroughputs();
		double fScoreRewardFromThroughputs();
		double fairnessFromThroughputs();
		double cumulatedThroughputFromThroughputs();
		std::vector<std::vector<double>> attainableThroughputs();
		std::vector<double> apThroughputsFromThroughputs();
		std::vector<double> staThroughputsFromThroughputs();
		std::vector<double> staPersFromPers();
		void computeThroughputsAndErrors();
		void setupNewConfiguration(NetworkConfiguration configuration);
		void endOfTest();
		static std::string configurationToString(const NetworkConfiguration& config);
		static int indexToChannelNumber(int i);
		static bool parameterConstraint(double sens, double pow);
		static unsigned int numberOfSamples(std::vector<GaussianT> gaussians);

	protected:
		pid_t _pid;
		Reward _rewardType;
		double _duration;
		double _testDuration;
		double _beta;
		double _windowSize;
		std::vector<double> _rewards;
		std::vector<double> _fairness;
		std::vector<double> _cumulatedThroughput;
		std::vector<NetworkConfiguration> _configurations;
		std::vector<std::vector<double>> _apThroughputs;
		std::vector<std::vector<double>> _staThroughputs;
		std::vector<std::vector<double>> _staPERs;
		std::vector<double> _positionAPX;
		std::vector<double> _positionAPY;
		std::vector<double> _positionStaX;
		std::vector<double> _positionStaY;
		std::vector<std::vector<double>> _throughputs;
		std::vector<std::vector<double>> _pers;
		std::vector<std::vector<unsigned int>> _associations;
		std::vector<NetDeviceContainer> _devices;
		NetworkConfiguration _configuration;
		Optimizer* _optimizer;
		std::vector<ApplicationContainer> _serversPerAp;
		std::vector<std::vector<unsigned int>> _lastRxPackets;
		std::vector<std::vector<unsigned int>> _lastLostPackets;
		double _interval;
		std::vector<double> _programSteps;
		std::vector<std::vector<std::vector<double>>> _attainableThroughputs;
		double _cumulatedTime = 0.0;
		unsigned int _packetSize = 1464;
		unsigned int _stepIndex = 0;
		int _defaultSensibility = -82;
		int _defaultPower = 20;
		double _attainableThroughput = 300.0e6;
};

#endif