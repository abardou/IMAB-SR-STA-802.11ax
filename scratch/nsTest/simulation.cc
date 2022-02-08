#include "simulation.hh"
#include <iostream>

/**
 * Create a simulation in a dedicated process
 * 
 * @param oId Optim the optimizer to use
 * @param sId Samp the sampler to use
 * @param r Reward the reward to use
 * @param topoPath std::string the path to the network topology
 * @param duration double the simulation duration
 * @param testDuration double the test duration
 * @param outputName std::string the output file name
 * @param beta double the beta parameter for FSCORE reward
 */ 
Simulation::Simulation(Optim oId, Samp sId, Reward r, std::string topoPath, double duration, double testDuration, std::string outputName, std::vector<double> programSteps, std::vector<double> saturationProgram, double beta) : _rewardType(r), _duration(duration), _testDuration(testDuration), _beta(beta), _programSteps(programSteps) {
	this->_pid = fork();
	if (this->_pid == 0) {
		// Child process

		// Topology
		this->readTopology(topoPath);

		this->_interval = 8.0 * this->_packetSize / this->_attainableThroughput;

		// Index of channel to use during the simulation
		unsigned int channel = 0, numberOfAPs = this->_positionAPX.size(), numberOfStas = 0;
		for (std::vector<unsigned int> assocs: this->_associations)
			numberOfStas += assocs.size();

		double applicationStart = 1.0, applicationEnd = applicationStart + duration;

		//Adapt interval as function of the number of stations
		std::vector<double> intervalsCross(numberOfAPs);
		for (unsigned int i = 0; i < numberOfAPs; i++) intervalsCross[i] = this->_interval * this->_associations[i].size();

		// APs creation and configuration
		// At the start, they're all configured with 802.11 default conf
		NodeContainer nodesAP;
		nodesAP.Create(numberOfAPs);
		std::vector<YansWifiPhyHelper> wifiPhy(numberOfAPs); // One PHY for each AP
		for(unsigned int i = 0; i < numberOfAPs; i++) {
			wifiPhy[i] = YansWifiPhyHelper::Default();
			wifiPhy[i].Set("Antennas", UintegerValue(2));
			// 2 spatial streams to support htMcs from 8 to 15 with short GI
			wifiPhy[i].Set("MaxSupportedTxSpatialStreams", UintegerValue(2));
			wifiPhy[i].Set("MaxSupportedRxSpatialStreams", UintegerValue(2));
			wifiPhy[i].Set("ChannelNumber", UintegerValue(channel));
			wifiPhy[i].Set("RxSensitivity", DoubleValue(this->_defaultSensibility));
			wifiPhy[i].Set("CcaEdThreshold", DoubleValue(this->_defaultSensibility));
			wifiPhy[i].Set("TxPowerStart", DoubleValue(this->_defaultPower));
			wifiPhy[i].Set("TxPowerEnd", DoubleValue(this->_defaultPower));
		}

		// Stations creation and configuration
		std::vector<NodeContainer> nodesSta(numberOfAPs);
		for(unsigned int i = 0; i < numberOfAPs; i++) nodesSta[i].Create(this->_associations[i].size());
		// One phy for every station
		YansWifiPhyHelper staWifiPhy = YansWifiPhyHelper::Default();
		staWifiPhy.Set("Antennas", UintegerValue(2));
		staWifiPhy.Set("MaxSupportedTxSpatialStreams", UintegerValue(2));
		staWifiPhy.Set("MaxSupportedRxSpatialStreams", UintegerValue(2));
		staWifiPhy.Set("ChannelNumber", UintegerValue(channel));
		staWifiPhy.Set("RxSensitivity", DoubleValue(this->_defaultSensibility));
		staWifiPhy.Set("CcaEdThreshold", DoubleValue(this->_defaultSensibility));

		// Propagation model, same for everyone
		YansWifiChannelHelper wifiChannel;
		wifiChannel.SetPropagationDelay ("ns3::ConstantSpeedPropagationDelayModel");
		wifiChannel.AddPropagationLoss ("ns3::LogDistancePropagationLossModel");
		Ptr<YansWifiChannel> channelPtr = wifiChannel.Create ();
		// Attribution to stations and models
		staWifiPhy.SetChannel(channelPtr);
		for(unsigned int i = 0; i < numberOfAPs; i++) wifiPhy[i].SetChannel(channelPtr);

		// 802.11ax protocol
		WifiHelper wifi;
		wifi.SetStandard (WIFI_PHY_STANDARD_80211ax_5GHZ);
		// TOCHANGE, Vht pour ax (0 pour le contrôle, à fixer pour les données)
		wifi.SetRemoteStationManager("ns3::ConstantRateWifiManager", "DataMode", StringValue ("VhtMcs4"), "ControlMode", StringValue ("VhtMcs0"));


		// Configure Infrastructure mode and SSID
		std::vector<NetDeviceContainer> devices(numberOfAPs);//Un groupe de Sta par AP
		Ssid ssid = Ssid ("ns380211");

		// Mac for Stations
		WifiMacHelper wifiMac;
		wifiMac.SetType("ns3::StaWifiMac", "Ssid", SsidValue(ssid));
		for (unsigned int i = 0; i < numberOfAPs; i++) devices[i] = wifi.Install(staWifiPhy, wifiMac, nodesSta[i]);
		// Mac for APs
		this->_devices = std::vector<NetDeviceContainer>(numberOfAPs);
		wifiMac.SetType("ns3::ApWifiMac", "Ssid", SsidValue(ssid));
		for (unsigned int i = 0; i < numberOfAPs; i++) this->_devices[i] = wifi.Install(wifiPhy[i], wifiMac, nodesAP.Get(i));

		// Mobility for devices
		MobilityHelper mobility;
		Ptr<ListPositionAllocator> positionAlloc = CreateObject<ListPositionAllocator>();
		// For APs
		for (unsigned int i = 0; i < numberOfAPs; i++) positionAlloc->Add(Vector(this->_positionAPX[i], this->_positionAPY[i], 0.0));
		// For stations
		for (unsigned int i = 0; i < numberOfAPs; i++)
			for (unsigned int j = 0 ; j < this->_associations[i].size(); j++)
				positionAlloc->Add(Vector(this->_positionStaX[this->_associations[i][j]], this->_positionStaY[this->_associations[i][j]], 0.0));
		// Devices are not moving
		mobility.SetPositionAllocator(positionAlloc);
		mobility.SetMobilityModel("ns3::ConstantPositionMobilityModel");
		// Application to APs
		mobility.Install(nodesAP);
		// Application to stations
		for(unsigned int i = 0; i < numberOfAPs; i++) mobility.Install(nodesSta[i]);

		//IP stack and addresses
		InternetStackHelper internet;
		for(unsigned int i = 0; i < numberOfAPs; i++) internet.Install(nodesSta[i]);
		internet.Install(nodesAP);

		Ipv4AddressHelper ipv4;
		ipv4.SetBase("10.1.0.0", "255.255.0.0");
		Ipv4InterfaceContainer apInterfaces;
		for(unsigned int i = 0; i < numberOfAPs; i++) {
			apInterfaces = ipv4.Assign(this->_devices[i]);
			ipv4.Assign(devices[i]);
		}

		// Traffic configuration
		// Server is installed on all stations
		uint16_t port = 4000;
		UdpServerHelper server(port);
		this->_serversPerAp = std::vector<ApplicationContainer>(numberOfAPs);
		for(unsigned int i = 0; i < numberOfAPs; i++) {
			ApplicationContainer apps = server.Install(nodesSta[i]);
			apps.Start(Seconds(applicationStart));
			apps.Stop(Seconds(applicationEnd));
			this->_serversPerAp[i] = apps;
		}
		// Client is installed on all APs
		this->_attainableThroughputs = std::vector<std::vector<std::vector<double>>>(programSteps.size(), std::vector<std::vector<double>>(numberOfAPs, std::vector<double>()));
		std::random_device rd;
		std::default_random_engine e2(rd()) ;
		std::uniform_real_distribution<> dist(0, 1);
		double nonSaturationRate = 0.01;
		for(unsigned int i = 0; i < numberOfAPs; i++) {
			// std::cout << "== AP " << i << " ==" << std::endl;
			for(unsigned int j = 0; j < this->_associations[i].size(); j++) {
				// std::cout << "\t* STA " << j << ":" << std::endl;
				// IPv4 instance of the station
				Ipv4Address addr = nodesSta[i].Get(j)
					->GetObject<Ipv4>()
					->GetAddress(1, 0) // Loopback (1-0)
					.GetLocal();

				for (unsigned int k = 0; k < programSteps.size(); k++) {
					double rate = intervalsCross[i] / (dist(e2) < saturationProgram[k] ? 1.0 : nonSaturationRate);
					this->_attainableThroughputs[k][i].push_back(8.0 * this->_packetSize / rate);
					UdpClientHelper clientCT;//CT=cross traffic (from AP to stations)
					clientCT.SetAttribute ("RemoteAddress",AddressValue(addr));
					clientCT.SetAttribute ("RemotePort",UintegerValue(port));
					clientCT.SetAttribute ("MaxPackets", UintegerValue(1e9));
					clientCT.SetAttribute ("Interval", TimeValue(Seconds(rate)));
					clientCT.SetAttribute ("PacketSize", UintegerValue(this->_packetSize));

					// Installation on AP
					ApplicationContainer apps = clientCT.Install(nodesAP.Get(i));
					// std::cout << "\t\tSTEP " << k << " from " << applicationStart + duration * programSteps[k] << "s to " << applicationStart + duration * ((k < programSteps.size() - 1) ? programSteps[k+1] : 1.0) << "s : " << 8 * this->_packetSize / (rate * 1e6) << " Mbps (" << rate << ")" << std::endl;
					apps.Start(Seconds(applicationStart + duration * programSteps[k]));
					apps.Stop(Seconds(applicationStart + duration * ((k < programSteps.size() - 1) ? programSteps[k+1] : 1.0)));
				}
			}
		}

		Simulator::Stop(Seconds(applicationEnd+0.01));

		// Optimization relative objects
		// Init callback for configuration changes
		for (unsigned int i = 0; i < numberOfAPs; i++) {
			this->_configuration.push_back(std::make_tuple(this->_defaultSensibility, this->_defaultPower));
		}
		Simulator::Schedule(Seconds(applicationStart+testDuration), &Simulation::endOfTest, this);

		// Init containers for throughput calculation
		for (unsigned int i = 0; i < numberOfAPs; i++) {
			unsigned int nStas = this->_associations[i].size();
			this->_throughputs.push_back(std::vector<double>(nStas, 0));
			this->_pers.push_back(std::vector<double>(nStas, 0));
			this->_lastRxPackets.push_back(std::vector<unsigned int>(nStas, 0));
			this->_lastLostPackets.push_back(std::vector<unsigned int>(nStas, 0));
		}

		// Parameters to optimize
		std::vector<ConstrainedCouple> parameters(numberOfAPs);
		for (unsigned int i = 0; i < numberOfAPs; i++) {
			parameters[i] = ConstrainedCouple(SingleParameter(-82, -62, 1), SingleParameter(1, 21, 1), &parameterConstraint);
		}

		// Sampler to use
		Sampler* sampler = nullptr;
		switch (sId) {
			case UNIF: sampler = new UniformSampler(parameters); break;
			case HGM: sampler = new HGMTSampler(parameters, this->_configuration, 6, 1.0 / (numberOfStas + 1.0), 1, &numberOfSamples); break;
		}

		// Optimizer to use
		switch (oId) {
			case EGREEDY: this->_optimizer = new EpsilonGreedyOptimizer(sampler, 0.09); break;
			case THOMP_BETA: this->_optimizer = new ThompsonBetaOptimizer(sampler, 2.4); break; // 2
			case THOMP_GAMNORM: this->_optimizer = new ThompsonGammaNormalOptimizer(sampler, 2, 2.4); break;
			case THOMP_NORM: this->_optimizer = new ThompsonNormalOptimizer(sampler, 2.4); break; 
		}

		std::cout << "ns3-debug: the simulation begins" << std::endl;
		Simulator::Run();

		Simulator::Destroy();

		// Stringstream for vector data
		std::ofstream myfile;
		myfile.open ("./scratch/nsTest/data/" + outputName);
		myfile << "rew,fair,cum,aps,stas,conf" << std::endl;
		for (unsigned int i = 0; i < this->_rewards.size(); i++) {
			std::stringstream aps, stas, pers;
			std::string delimiter = ",";
			copy(this->_apThroughputs[i].begin(), this->_apThroughputs[i].end(), std::ostream_iterator<double>(aps, delimiter.c_str()));
			copy(this->_staThroughputs[i].begin(), this->_staThroughputs[i].end(), std::ostream_iterator<double>(stas, delimiter.c_str()));
			copy(this->_staPERs[i].begin(), this->_staPERs[i].end(), std::ostream_iterator<double>(pers, delimiter.c_str()));

			std::string apsData = aps.str();
			std::string stasData = stas.str();
			std::string persData = pers.str();
			apsData = apsData.substr(0, apsData.size() - 1);
			stasData = stasData.substr(0, stasData.size() - 1);
			persData = persData.substr(0, persData.size() - 1);

			myfile << this->_rewards[i] << "\t" << this->_fairness[i] << "\t" << this->_cumulatedThroughput[i] << "\t"
						 << apsData << "\t" << stasData << "\t" << persData << "\t"
						 << this->configurationToString(this->_configurations[i]) << std::endl;
		}
		myfile.close();

		// Free sampler and optimizer
		delete sampler;
		delete this->_optimizer;
		
		exit(0);
	}
}

/**
 * Return the PID of the simulation
 * 
 * @return the PID of the simulation
 */
pid_t Simulation::getPID() const {
	return this->_pid;
}

/**
 * Compute the adequate reward from throughputs of STAs
 * 
 * @return the adequate reward computation
 */
double Simulation::rewardFromThroughputs() {
	if (this->_rewardType == AD_HOC)
		return this->adHocRewardFromThroughputs();
	return this->fScoreRewardFromThroughputs();
}

/**
 * This function is called at each end of test
 */
void Simulation::endOfTest() {
	this->_cumulatedTime += this->_testDuration;
	bool found = false;
	for (unsigned int i = 1; i < this->_programSteps.size(); i++) {
		if (this->_programSteps[i] * this->_duration > this->_cumulatedTime) {
			this->_stepIndex = i-1;
			found = true;
			break;
		}
	}
	if (!found) this->_stepIndex = this->_programSteps.size() - 1;

  // Compute the throughput
  this->computeThroughputsAndErrors();
	// Store metrics
	this->storeMetrics();
  // Compute reward accordingly
  double rew = this->rewardFromThroughputs();

  // Add config and reward to sampler
  this->_optimizer->addToBase(this->_configuration, rew);

  // Use the optimizer to get another configuration
  NetworkConfiguration configuration = this->_optimizer->optimize();
  // Set up the new configuration
  setupNewConfiguration(configuration);

  // Next scheduling for recurrent callback
  Simulator::Schedule(Seconds(this->_testDuration), &Simulation::endOfTest, this);
}

/**
 * Store the network metrics in dedicated containers.
 * This method should be called AFTER computeThroughputsAndErrors.
 */
void Simulation::storeMetrics() {
	this->_configurations.push_back(this->_configuration);

	double rew = this->rewardFromThroughputs();
	this->_rewards.push_back(rew);
	// std::cout << "Reward: " << this->rewardFromThroughputs() << " vs. " << this->otherRewardFromThroughputs() << std::endl;

	double fairness = this->fairnessFromThroughputs();
	this->_fairness.push_back(fairness);
	// std::cout << "Fairness: " << fairness << std::endl;

	double cumThrough = this->cumulatedThroughputFromThroughputs();
	this->_cumulatedThroughput.push_back(cumThrough);
	// std::cout << "CumThrough: " << cumThrough << std::endl << std::endl;

	this->_apThroughputs.push_back(this->apThroughputsFromThroughputs());
	this->_staThroughputs.push_back(this->staThroughputsFromThroughputs());
	this->_staPERs.push_back(this->staPersFromPers());
}

/**
 * Compute the reward to assess the quality of a test.
 * An AP is seen as starving if its throughput is less than 10 % of its
 * attainable throughput.
 * 
 * @return the reward 
 */
double Simulation::adHocRewardFromThroughputs() {
  double starvRew = 1, noStarvRew = 1, nStarv = 0, nNoStarv = 0;
  for (unsigned int i = 0; i < this->_throughputs.size(); i++)
    for (unsigned int j = 0; j < this->_throughputs[i].size(); j++) {
      double threshold = 0.1 * this->_attainableThroughputs[this->_stepIndex][i][j];
      if (this->_throughputs[i][j] < threshold) {
        this->_throughputs[i][j] = std::max(this->_throughputs[i][j], 1.0);
        starvRew *= this->_throughputs[i][j] / threshold;
				nStarv++;
      } else {
				noStarvRew *= this->_throughputs[i][j] / this->_attainableThroughputs[this->_stepIndex][i][j];
				nNoStarv++;
			}
    }
		
  double n = nStarv + nNoStarv;
  // Compute global reward
  return (nStarv * starvRew + nNoStarv * (noStarvRew + n)) / (n * (n + 1.0));
}

/**
 * Compute the fscore reward for a test
 * 
 * @return the fscore reward
 */
double Simulation::fScoreRewardFromThroughputs() {
	double fairness = this->fairnessFromThroughputs();
  double sum = 0, maxThroughput = 0;
  for (unsigned int i = 0; i < this->_throughputs.size(); i++)
    for (unsigned int j = 0; j < this->_throughputs[i].size(); j++) {
      sum += this->_throughputs[i][j] / 1.0e6;
			maxThroughput += this->_attainableThroughputs[this->_stepIndex][i][j] / 1.0e6;
    }
	double cumThrough = sum / maxThroughput;

	return (1.0 + this->_beta * this->_beta) * fairness * cumThrough / (fairness + this->_beta * this->_beta * cumThrough);
}

/**
 * Compute the fairness of the network (Jain's index).
 * 
 * @return the fairness 
 */
double Simulation::fairnessFromThroughputs() {
  double squareOfMean = 0, meanOfSquares = 0;
	unsigned int n = 0;
  for (unsigned int i = 0; i < this->_throughputs.size(); i++)
    for (unsigned int j = 0; j < this->_throughputs[i].size(); j++) {
			double tMbps = this->_throughputs[i][j] / 1.0e6;
      squareOfMean += tMbps;
			meanOfSquares += tMbps * tMbps;
			n++;
    }

	squareOfMean *= squareOfMean / (n * n);
	meanOfSquares /= n;

	return squareOfMean / meanOfSquares;
}

/**
 * Compute the cumulated throughput.
 * 
 * @return the cumulated throughput 
 */
double Simulation::cumulatedThroughputFromThroughputs() {
  double cumThroughput = 0;
  for (unsigned int i = 0; i < this->_throughputs.size(); i++)
    for (unsigned int j = 0; j < this->_throughputs[i].size(); j++)
			cumThroughput += this->_throughputs[i][j];

	return cumThroughput;
}

/**
 * Compute the throughput of each AP.
 * 
 * @return the throughput of each AP 
 */
std::vector<double> Simulation::apThroughputsFromThroughputs() {
  std::vector<double> apThroughputs;
  for (unsigned int i = 0; i < this->_throughputs.size(); i++) {
		double apThroughput = 0;
		for (unsigned int j = 0; j < this->_throughputs[i].size(); j++)
			apThroughput += this->_throughputs[i][j];
		apThroughputs.push_back(apThroughput);
	}

	return apThroughputs;
}

/**
 * Compute the throughput of each STA.
 * 
 * @return the throughput of each STA 
 */
std::vector<double> Simulation::staThroughputsFromThroughputs() {
  std::vector<double> staThroughputs;
  for (unsigned int i = 0; i < this->_throughputs.size(); i++)
		for (unsigned int j = 0; j < this->_throughputs[i].size(); j++)
			staThroughputs.push_back(this->_throughputs[i][j]);

	return staThroughputs;
}

/**
 * Compute the PER of each STA.
 * 
 * @return the PER of each STA 
 */
std::vector<double> Simulation::staPersFromPers() {
  std::vector<double> staPers;
  for (unsigned int i = 0; i < this->_pers.size(); i++)
		for (unsigned int j = 0; j < this->_pers[i].size(); j++)
			staPers.push_back(this->_pers[i][j]);

	return staPers;
}

/**
 * Compute the throughput of each station
 */
void Simulation::computeThroughputsAndErrors() {
  // Compute throughput for each server app
	// double overall = 0;
  for (unsigned int i = 0; i < this->_serversPerAp.size(); i++) {
    std::vector<double> staAPThroughputs(this->_lastRxPackets[i].size());
		std::vector<double> staPERs(this->_lastLostPackets[i].size());
    for (unsigned int j = 0; j < this->_serversPerAp[i].GetN(); j++) {
      // Received bytes since the start of the simulation
			UdpServer* server = dynamic_cast<UdpServer*>(GetPointer(this->_serversPerAp[i].Get(j)));
			unsigned int lostPackets = server->GetLost();
			unsigned int testLostPackets = lostPackets - this->_lastLostPackets[i][j];
			unsigned int receivedPackets = server->GetReceived();
			unsigned int testReceivedPackets = receivedPackets - this->_lastRxPackets[i][j];
			double per = testLostPackets + testReceivedPackets != 0 ? ((double) testLostPackets) / (testLostPackets + testReceivedPackets) : 0;

			// std::cout << testLostPackets << " and " << testReceivedPackets << " => " << per << std::endl;

      double rxBytes = this->_packetSize * testReceivedPackets;
      // Compute the throughput considering only unseen bytes
      double throughput = std::min(rxBytes * 8.0 / this->_testDuration, this->_attainableThroughputs[this->_stepIndex][i][j]); // bit/s, std::min fixes bad approximations for short test periods
      // Update containers
      staAPThroughputs[j] = throughput;
			staPERs[j] = per;
      this->_lastRxPackets[i][j] = receivedPackets;
			this->_lastLostPackets[i][j] = lostPackets;
			// overall += throughput / 1e6;
      // Log
      // std::cout << "Station " << j << " of AP " << i << " : " << (throughput / 1e6) << " Mbit/s" << std::endl;
    }
    this->_throughputs[i] = staAPThroughputs;
		this->_pers[i] = staPERs;
  }

	// std::cout << "OVERALL: " << overall << " Mbit/s" << std::endl;
}

/**
 * Set up a new configuration in the simulation
 * 
 * @param configuration the configuration to set
 */ 
void Simulation::setupNewConfiguration(NetworkConfiguration configuration) {
  int nNodes = this->_devices.size();
  for (int i = 0; i < nNodes; i++) {
    Ptr<WifiPhy> phy = dynamic_cast<WifiNetDevice*>(GetPointer((this->_devices[i].Get(0))))->GetPhy();

    double sensibility = std::get<0>(configuration[i]),
           txPower = std::get<1>(configuration[i]);
    phy->SetTxPowerEnd(txPower);
    phy->SetTxPowerStart(txPower);
    phy->SetRxSensitivity(sensibility);
    phy->SetCcaEdThreshold(sensibility);
  }

	this->_configuration = configuration;
}

/**
 * Read topology from a JSON file.
 * 
 * @param path string the path of the JSON file
 */
void Simulation::readTopology(std::string path) {
	std::ifstream jsonFile(path);
  std::string jsonText((std::istreambuf_iterator<char>(jsonFile)), std::istreambuf_iterator<char>());

  Json::CharReaderBuilder readerBuilder;
  Json::CharReader* reader = readerBuilder.newCharReader();
  Json::Value topo;
  std::string errors;
  reader->parse(jsonText.c_str(), jsonText.c_str() + jsonText.size(), &topo, &errors);
  
  // APs
  for (Json::Value ap: topo["aps"]) {
    this->_positionAPX.push_back(ap["x"].asDouble());
    this->_positionAPY.push_back(ap["y"].asDouble());

    std::vector<unsigned int> assoc;
    for (Json::Value sta: ap["stas"]) {
      assoc.push_back(sta.asUInt());
    }
    this->_associations.push_back(assoc);
  }

  // Stations
  for (Json::Value sta: topo["stations"]) {
    this->_positionStaX.push_back(sta["x"].asDouble());
    this->_positionStaY.push_back(sta["y"].asDouble());
  }
}

/**
 * Boolean constraint for parameter combination
 * 
 * @param sens double the sensibility
 * @param pow double the transmission power
 * 
 * @return true if the constraint is validated, false otherwise 
 */
bool Simulation::parameterConstraint(double sens, double pow) {
  return sens <= std::max(-82.0, std::min(-62.0, -82.0 + 20.0 - pow));
}

/**
 * Compute the number of samples for a given gaussian mixture
 * 
 * @param gaussians Container the gaussian mixture
 * 
 * @return the number of tests to do before updating the mixture
 */
unsigned int Simulation::numberOfSamples(std::vector<GaussianT> gaussians) {
  double s = 0;
  for (GaussianT g: gaussians) {
		s += 0.5 * 2.0 * std::get<0>(g).size() * std::get<1>(g) / 0.05;
	}

  return round(s);
}

/**
 * Map channel index to a real channel.
 *
 * Channel number must be in {36, 40, etc.} for the 5GHz band.
 * The web page: https://www.nsnam.org/docs/models/html/wifi-user.html
 * Read in particular the WifiPhy::ChannelNumber section
 *
 * @param i int the index to map
 *
 * @return a channel corresponding to the mapped index
 */
int Simulation::indexToChannelNumber(int i) {
  switch (i) {
    case 0: return 42;
    case 1: return 58;
    case 2: return 106;
    case 3: return 122;
    case 4: return 138;
    case 5: return 155;
    default:
      std::cerr << "Error indexToChannelNumber(): the index is negative, null, or greater than the number of channels (12 - 40MHz)." << std::endl;
      return 42;
  }
}

/**
 * Turn a network configuration to a convenient string representation
 * 
 * @param config Container the configuration
 * 
 * @return a convenient representation of the configuration
 */
std::string Simulation::configurationToString(const NetworkConfiguration& config) {
	std::string result = "";
	for (unsigned int i = 0; i < config.size(); i++) {
		result += "(" + std::to_string(std::get<0>(config[i])) + "," + std::to_string(std::get<1>(config[i])) + ")";
		if (i < config.size() - 1)
			result += ",";
	}

	return result;
}