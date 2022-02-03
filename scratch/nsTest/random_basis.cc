// #include "ns3/core-module.h"
// #include "ns3/network-module.h"
// #include "ns3/mobility-module.h"
// #include "ns3/config-store-module.h"
// #include "ns3/wifi-module.h"
// #include "ns3/internet-module.h"
// #include "ns3/applications-module.h"

// #include<sys/wait.h>

// #include "json.hh"
// #include "simulation.hh"

// #include <cmath>
// #include <random>
// #include <chrono>

// using namespace ns3;

// /**
//  * Extract data from a simulation file in .tsv format and return it
//  * 
//  * @param path the path of the simulation file
//  * 
//  * @return the data contained in the simulation file
//  */
// std::vector<std::vector<std::string>> extractTSVData(std::string path) {
//   unsigned int nCols = 7;
//   std::vector<std::vector<std::string>> data(nCols);
//   std::ifstream myfile;
//   myfile.open(path);

//   std::string line, word;
//   // Don't care about the first line
//   getline(myfile, line);
//   while (getline(myfile, line)) {
//     // Retrieve the line and store it, splitting it first by \t
//     std::stringstream s(line);
//     for (unsigned int i = 0; i < nCols; i++) {
//       getline(s, word, '\t');
//       data[i].push_back(word);
//     }
//   }
//   myfile.close();

//   return data;
// }

// void aggregateSimulationsResults(std::string path, std::vector<std::string> names) {
//   std::ofstream resFile;
//   resFile.open(path+"_results.tsv");

//   // First line of aggregation
//   std::string delimiter = "\t";
//   std::string firstLine = "";

//   firstLine = "x" + delimiter + "y" + delimiter + "rew" + delimiter + "conf";
//   resFile << firstLine << std::endl;

//   // Each line is a simulation
// 	unsigned int i = 0, n = names.size(), middle = n / 2;
//   for (std::string name: names) {
// 		int x = i - middle;
//     std::vector<std::vector<std::string>> data = extractTSVData(name);

// 		unsigned int size = data[0].size();
// 		unsigned int capped_size = std::min((unsigned int) data[0].size(), 2 * middle);
// 		for (unsigned int j = 1; j < size; j++) {
// 			int y = (x <= 0) ? middle - capped_size + j : -middle + j;
// 			resFile << x << delimiter << y << delimiter << data[0][j] << delimiter << data[6][j] << std::endl;
// 		}

// 		i++;

//     // Remove the data file
//     remove(name.c_str());
//   }

//   resFile.close();
// }

// /**
//  * Return a proper string representing the double passed in parameter
//  * 
//  * @param d double to transform in string
//  * 
//  * @return a string representing the double
//  */
// std::string doubleToString(double d) {
//   std::string str = std::to_string(d);
//   str.erase(str.find_last_not_of('0') + 1, std::string::npos);

//   return str.back() == '.' ? str + '0' : str;
// }

// /**
//  * Read topology from a JSON file.
//  * 
//  * @param path string the path of the JSON file
//  * 
//  * @return the JSON representation of the topology
//  */
// Json::Value readTopology(std::string path) {
// 	std::ifstream jsonFile(path);
//   std::string jsonText((std::istreambuf_iterator<char>(jsonFile)), std::istreambuf_iterator<char>());

//   Json::CharReaderBuilder readerBuilder;
//   Json::CharReader* reader = readerBuilder.newCharReader();
//   Json::Value topo;
//   std::string errors;
//   reader->parse(jsonText.c_str(), jsonText.c_str() + jsonText.size(), &topo, &errors);

//   return topo;
// }

// void readTopology(std::string path, std::vector<double>& positionAPX, std::vector<double>& positionAPY, std::vector<double>& positionAPZ, std::vector<double>& positionStaX, std::vector<double>& positionStaY, std::vector<double>& positionStaZ, std::vector<std::vector<unsigned int>>& associations) {
// 	std::ifstream jsonFile(path);
//   std::string jsonText((std::istreambuf_iterator<char>(jsonFile)), std::istreambuf_iterator<char>());

//   Json::CharReaderBuilder readerBuilder;
//   Json::CharReader* reader = readerBuilder.newCharReader();
//   Json::Value topo;
//   std::string errors;
//   reader->parse(jsonText.c_str(), jsonText.c_str() + jsonText.size(), &topo, &errors);
  
//   // APs
//   for (Json::Value ap: topo["aps"]) {
//     positionAPX.push_back(ap["x"].asDouble());
//     positionAPY.push_back(ap["y"].asDouble());
// 		positionAPZ.push_back(ap["z"].asDouble());
//     std::vector<unsigned int> assoc;
//     for (Json::Value sta: ap["stas"]) {
//       assoc.push_back(sta.asUInt());
//     }
//     associations.push_back(assoc);
//   }

//   // Stations
//   for (Json::Value sta: topo["stations"]) {
//     positionStaX.push_back(sta["x"].asDouble());
//     positionStaY.push_back(sta["y"].asDouble());
//     positionStaZ.push_back(sta["z"].asDouble());
//   }
// }

// double normalize(double x, double m, double M) {
// 	return (x - m) / (M - m);
// }

// double denormalize(double y, double m, double M) {
// 	return y * (M - m) + m;
// }

// int main(int argc, char *argv[]) {
// 	std::vector<double> positionAPX, positionAPY, positionAPZ, positionStaX, positionStaY, positionStaZ;
// 	std::vector<std::vector<unsigned int>> associations;

// 	// Settings
// 	std::string topoName = "HLM"; // "MER_FLOORS_CH20_S5";
// 	readTopology("./scratch/nsTest/topos/" + topoName + ".json", positionAPX, positionAPY, positionAPZ, positionStaX, positionStaY, positionStaZ, associations);
// 	unsigned int N = 400,
// 							 D = 2 * positionAPX.size();
// 	double sensMin = -82, sensMax = -62,
// 				 powMin = 1, powMax = 21,
// 				 sqrtN2 = sqrt(N) / 2.0;
// 	std::vector<double> center({-76, 9, -76, 11, -67, 3, -70, 8, -75, 9, -74, 8, -71, 5, -66, 6, -73, 9, -68, 10, -65, 3, -66, 12, -78, 6, -76, 16}),
// 	// std::vector<double> center({-73, 10, -74, 10, -73, 12, -77, 9, -76, 14, -72, 12, -72, 10, -74, 10, -74, 13, -72, 10}),
// 											centerNormalized(D, 0),
// 											xbase(D, 0), ybase(D, 0),
// 											x(D, 0), y(D, 0);
// 	// Random vectorial basis
// 	std::default_random_engine generator = std::default_random_engine(std::chrono::system_clock::now().time_since_epoch().count());
// 	std::uniform_int_distribution<> sens(sensMin, sensMax), pow(powMin, powMax);
// 	for (unsigned int i = 0; i < D; i++) {
// 		if (i % 2 == 0) {
// 			centerNormalized[i] = normalize(center[i], sensMin, sensMax);
// 			x[i] = sens(generator);
// 			y[i] = sens(generator);
// 			xbase[i] = normalize(x[i], sensMin, sensMax);
// 			ybase[i] = normalize(y[i], sensMin, sensMax);
// 		} else {
// 			centerNormalized[i] = normalize(center[i], powMin, powMax);
// 			x[i] = pow(generator);
// 			y[i] = pow(generator);
// 			xbase[i] = normalize(x[i], powMin, powMax);
// 			ybase[i] = normalize(y[i], powMin, powMax);
// 		}
// 	}

// 	// c + xb = l, (l - c) / b = x
// 	double dx = 1000000, dy = 1000000,
// 				 xbound, ybound;
// 	unsigned int nConfigs = 0;
// 	for (unsigned int i = 0; i < D; i++) {
// 		xbound = std::min(xbase[i] == 0 ? 1000000000 : abs((0 - centerNormalized[i]) / xbase[i]), xbase[i] == 0 ? 1000000000 : abs((1 - centerNormalized[i]) / xbase[i]));
// 		ybound = std::min(ybase[i] == 0 ? 1000000000 : abs((0 - centerNormalized[i]) / ybase[i]), ybase[i] == 0 ? 1000000000 : abs((1 - centerNormalized[i]) / ybase[i]));

// 		dx = std::min(dx, xbound);
// 		dy = std::min(dy, ybound);
// 	}
// 	dx /= sqrtN2;
// 	dy /= sqrtN2;

// 	std::cout << "x: (";
// 	unsigned int i = 0;
// 	for (double xi: xbase) {
// 		std::cout << ", " << ((i % 2 == 0) ? "-" : "") << round(100 * 21 * dx * xi) / 100.0;
// 		i++;
// 	}
// 	i = 0;
// 	std::cout << std::endl << "y: (";
// 	for (double yi: ybase) {
// 		std::cout << ", " << ((i % 2 == 0) ? "-" : "") << round(100 * 21 * dy * yi) / 100.0;
// 		i++;
// 	}
// 	std::cout << std::endl;

// 	double minI, maxI;
// 	std::vector<std::vector<NetworkConfiguration>> confs;
// 	for (int x = -sqrtN2; x <= sqrtN2; x++) {
// 		std::vector<NetworkConfiguration> sim_confs;
// 		for (int y = -sqrtN2; y <= sqrtN2; y++) {
// 			std::vector<double> c(D, 0);
// 			for (unsigned int i = 0; i < D; i++) {
// 				if (i % 2 == 0) {
// 					minI = sensMin;
// 					maxI = sensMax;
// 				} else {
// 					minI = powMin;
// 					maxI = powMax;
// 				}
// 				c[i] = std::max(std::min(centerNormalized[i] + x * dx * xbase[i] + y * dy * ybase[i], 1.0), 0.0);
// 				c[i] = round(denormalize(c[i], minI, maxI));
// 			}
// 			NetworkConfiguration nc;
// 			for (unsigned int i = 0; i < D; i += 2)
// 				nc.push_back(std::make_tuple(c[i], c[i+1]));

// 			sim_confs.push_back(nc);
// 			nConfigs++;
// 		}

// 		confs.push_back(sim_confs);
// 	}	

// 	std::cout << nConfigs << " configurations to test on " << topoName << std::endl;
// 	double testDuration = 0.375; // 0.375;
// 	bool uplink = true;

// 	std::vector<Simulation*> simulations(2 * sqrtN2);
// 	std::vector<std::string> names;
// 	for (unsigned int i = 0; i < 2 * sqrtN2; i++) {
// 		unsigned int nConfigs = confs[i].size();
// 		double duration = testDuration * (nConfigs - 2);
// 		Optim o = LIST;
// 		Samp s = UNIF;
// 		ChannelWidth cw = MHZ_20;
// 		Entry entry = DEF;
// 		DistanceMode dm = STD;
// 		Reward rew = LOGPF;
// 		// Station throughput
// 		std::vector<std::vector<unsigned int>> predef_throughputs = {};
// 		std::vector<double> stations_types({0.0, 0, 0, 1.0});

// 		std::discrete_distribution<int> discreteDist(stations_types.begin(), stations_types.end());
// 		std::default_random_engine generator(std::chrono::system_clock::now().time_since_epoch().count());
// 		// Reading the topology object to get the number of stations
// 		Json::Value topology = readTopology("./scratch/nsTest/topos/" + topoName + ".json");
		
// 		// Attribute stations types
// 		std::vector<StationThroughput> stations_throughputs;
// 		for (Json::Value sta : topology["stations"]) {
// 			int st = discreteDist(generator);
// 			stations_throughputs.push_back(static_cast<StationThroughput>(st));
// 		}
// 		std::string stId = "";
// 		for (StationThroughput sta : stations_throughputs) {
// 			stId += std::to_string(sta);
// 		}

// 		// Find the right identifiers
// 		std::string oId = "", sId = "", rId = "", eId = "", dId = "", cId = "";
// 		switch (o) {
// 			case IDLEOPT: oId = "IDLE"; break;
// 			case THOMP_GAMNORM: oId = "TGNORM"; break;
// 			case THOMP_BETA: oId = "TBETA"; break;
// 			case THOMP_NORM: oId = "TNORM"; break;
// 			case EGREEDY: oId = "EGREED"; break;
// 			case MARGIN: oId = "MARGIN"; break;
// 			case GP: oId = "GP"; break;
// 			case MULTIGP: oId = "MULTIGP"; break;
// 			case MULTIJNCA19: oId = "MULTIJNCA"; break;
// 			case LIST: oId = "LIST"; break;
// 		}

// 		switch (s) {
// 			case HGM: sId = "HGMT"; break;
// 			case UNIF: sId = "UNI"; break;
// 			case HCM: sId = "HCM"; break;
// 			case HBLAS: sId = "HBLAS"; break;
// 		}

// 		switch (rew) {
// 			case AD_HOC: rId = "ADHOC"; break;
// 			case CUMTP: rId = "CUMTP"; break;
// 			case LOGPF: rId = "LOGPF"; break;
// 		}

// 		switch (entry) {
// 			case DEF: eId = "DEF"; break;
// 			case CONV: eId = "CONV"; break;
// 			case OPP: eId = "OPP"; break;
// 			case CONV_MER: eId = "CONVMER"; break;
// 			case DIAG: eId = "DIAG"; break;
// 			case DEGNH: eId = "DEGNH"; break;
// 			case DEGA: eId = "DEGA"; break;
// 		}

// 		switch (dm) {
// 			case STD: dId = "STD"; break;
// 			case CYCLE: dId = "CYCLE"; break;
// 		}

// 		switch (cw) {
// 			case MHZ_20: cId = "20"; break;
// 			case MHZ_40: cId = "40"; break;
// 			case MHZ_80: cId = "80"; break;
// 		}

// 		BuildingInfo bi = {topoName == "HLM", 6, 4, 9, 5, 5, 2.5, 0.3};

// 		// Build the template of the output
// 		std::string traffic_type = uplink ? "BOTH" : "DOWN";
// 		std::string outputName = topoName + "_" + stId + "_" + cId + "_" + eId + "_" + dId + "_" + doubleToString(duration) + "_" + oId + "_" + sId + "_" + rId + "_" + doubleToString(testDuration) + "_" + traffic_type;
// 		// std::string outputName = topo + "_" + stId + deg + cId + "_" + eId + "_" + dId + "_" + doubleToString(duration) + "_0.9_" + doubleToString(testDuration);

// 		std::cout << outputName + "_" + std::to_string(i) + ".tsv" << std::endl;

// 		// Launching simulation
// 		usleep(1000);
// 		simulations[i] = new Simulation(o, s, rew, entry, dm, cw, topology, stations_throughputs, duration, testDuration, uplink, outputName + "_" + std::to_string(i) + ".tsv", {}, confs[i], bi);
// 		names.push_back("./scratch/nsTest/data/" + outputName + "_" + std::to_string(i) + ".tsv");
// 	}

// 	// Wait for all simulations to finish
// 	for (unsigned int i = 0; i < 2 * sqrtN2; i++) {
// 		waitpid(simulations[i]->getPID(), NULL, 0);
// 		delete simulations[i];
// 	}
// 	std::cout << "Simulations terminated" << std::endl;

// 	// Aggregate the data
// 	aggregateSimulationsResults("./scratch/nsTest/data/"+topoName+"_RandomBasis", names);
// 	std::cout << "Aggregation terminated" << std::endl << std::endl;
	
// }