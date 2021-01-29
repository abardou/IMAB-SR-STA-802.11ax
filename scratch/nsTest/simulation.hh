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

enum Optim { EGREEDY, THOMP_BETA, THOMP_GAMNORM, THOMP_NORM };
enum Samp { UNIF, HGM };
enum Reward { AD_HOC, FSCORE };
enum Dist { LOG, SQRT, N2, N4 };

class Simulation {
	public:
		Simulation(Optim oId, Samp sId, Reward r, std::string topoPath, double duration, double testDuration, std::string outputName, double beta=1.0);
		pid_t getPID() const;
		void readTopology(std::string path);
		void storeMetrics();
		double rewardFromThroughputs();
		double adHocRewardFromThroughputs();
		double fScoreRewardFromThroughputs();
		double fairnessFromThroughputs();
		double cumulatedThroughputFromThroughputs();
		NetworkConfiguration handleClusterizedConfiguration(const NetworkConfiguration& configuration);
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
		double _testDuration;
		double _beta;
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
		std::vector<unsigned int> _clustersAP;
		std::vector<std::vector<double>> _throughputs;
		std::vector<std::vector<double>> _pers;
		std::vector<std::vector<double>> _attainableThroughputs;
		std::vector<std::vector<unsigned int>> _associations;
		std::vector<NetDeviceContainer> _devices;
		NetworkConfiguration _configuration;
		Optimizer* _optimizer;
		std::vector<ApplicationContainer> _serversPerAp;
		std::vector<std::vector<unsigned int>> _lastRxPackets;
		std::vector<std::vector<unsigned int>> _lastLostPackets;
		unsigned int _packetSize = 1464;
		int _defaultSensibility = -82;
		int _defaultPower = 20;
		double _intervalCross = 0.00001;
		double _attainableThroughput = 300.0e6;
};

#endif