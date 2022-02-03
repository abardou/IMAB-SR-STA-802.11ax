#include <iostream>
#include <string>
#include <vector>
#include <random>

#include<sys/wait.h>

#include "json.hh"

#include "simulation.hh"

/**
 * Return a proper string representing the double passed in parameter
 * 
 * @param d double to transform in string
 * 
 * @return a string representing the double
 */
std::string doubleToString(double d) {
  std::string str = std::to_string(d);
  str.erase(str.find_last_not_of('0') + 1, std::string::npos);

  return str.back() == '.' ? str + '0' : str;
}

/**
 * Extract data from a simulation file in .tsv format and return it
 * 
 * @param path the path of the simulation file
 * 
 * @return the data contained in the simulation file
 */
std::vector<std::vector<std::string>> extractTSVData(std::string path) {
  unsigned int nCols = 7;
  std::vector<std::vector<std::string>> data(nCols);
  std::ifstream myfile;
  myfile.open(path);

  std::string line, word;
  // Don't care about the first line
  getline(myfile, line);
  while (getline(myfile, line)) {
    // Retrieve the line and store it, splitting it first by \t
    std::stringstream s(line);
    for (unsigned int i = 0; i < nCols; i++) {
      getline(s, word, '\t');
      data[i].push_back(word);
    }
  }
  myfile.close();

  return data;
}

/**
 * Write some data into a file given a delimiter
 * 
 * @param data Container the data to write
 * @param file std::ofstream& the file to write in
 * @param delimiter string the delimiter between each piece of data
 */
void writeInFile(const std::vector<std::string>& data, std::ofstream& file, const std::string& delimiter) {
  std::stringstream s;
  copy(data.begin(), data.end(), std::ostream_iterator<std::string>(s, delimiter.c_str()));
  std::string content = s.str();
  content = content.substr(0, content.size() - 1);
  file << content << std::endl;
}

/**
 * Gather all the simulation result into a single file
 * 
 * @param path string the path of the file to write the aggregated result
 * @param temp string the template of the simulation files
 * @param nSim unsigned int the number of simulation files
 * @param nTests unsigned int the number of tests in a single simulation
 */
void aggregateSimulationsResults(std::string path, std::string temp, unsigned int nSim, unsigned int nTests) {
  std::ofstream rewFile, fairFile, cumFile, apsFile, stasFile, confFile, persFile;
  rewFile.open(path+"_rew.tsv");
  fairFile.open(path+"_fair.tsv");
  cumFile.open(path+"_cum.tsv");
  apsFile.open(path+"_aps.tsv");
  stasFile.open(path+"_stas.tsv");
  persFile.open(path+"_pers.tsv");
  confFile.open(path+"_conf.tsv");

  // First line of aggregation
  std::string delimiter = "\t";
  std::string firstLine = "";
  for (unsigned int i = 0; i < nTests; i++) firstLine += std::to_string(i) + (i < nTests - 1 ? delimiter : "");
  rewFile << firstLine << std::endl;
  fairFile << firstLine << std::endl;
  cumFile << firstLine << std::endl;
  apsFile << firstLine << std::endl;
  stasFile << firstLine << std::endl;
  persFile << firstLine << std::endl;
  confFile << firstLine << std::endl;

  // Each line is a simulation
  for (unsigned int i = 0; i < nSim; i++) {
    std::vector<std::vector<std::string>> data = extractTSVData(temp + std::to_string(i) + ".tsv");
    writeInFile(data[0], rewFile, delimiter); // Reward
    writeInFile(data[1], fairFile, delimiter); // Fairness
    writeInFile(data[2], cumFile, delimiter); // Cum
    writeInFile(data[3], apsFile, delimiter); // APs
    writeInFile(data[4], stasFile, delimiter); // STAs
    writeInFile(data[5], persFile, delimiter); // PERs
    writeInFile(data[6], confFile, delimiter); // Configurations   

    // Remove the data file
    remove((temp + std::to_string(i) + ".tsv").c_str());
  }

  rewFile.close();
  fairFile.close();
  cumFile.close();
  apsFile.close();
  stasFile.close();
  confFile.close();
}

int main (int argc, char *argv[]) {
  // Number of simulations to run
  unsigned int nSimulations = 25;
  // Duration of a single simulation
  double duration = 120.0;
  // Duration of a single test
  double testDuration = 0.05;
  // Optimizers to test
  std::vector<Optim> optimizers({THOMP_GAMNORM, THOMP_NORM, EGREEDY});
  // Samplers to test
  std::vector<Samp> samplers({HGM, UNIF});
  // Rewards to test
  std::vector<Reward> rewards({AD_HOC});
  // Topos to test
  std::vector<std::string> topos({"T12"});
  // For each topo

  for (std::string topo: topos) {
    // Build a saturation regime
    std::vector<double> times({0, 1.0/3.0, 2.0/3.0});
    std::vector<double> saturatedStas({2.0/3.0, 1.0/3.0, 1.0});
    // For each optimizer
    for (Optim o: optimizers) {
      // For each sampler
      for (Samp s: samplers) {
        // Don't test some combinations
        if ((s == UNIF && (o == THOMP_GAMNORM)) || (s == HGM && o == EGREEDY)) {
          continue;
        }
        // For each reward
        for (Reward r: rewards) {
          // Find the right identifiers
          std::string oId = "", sId = "", rId = "";
          switch (o) {
            case THOMP_GAMNORM: oId = "TGNORM"; break;
            case THOMP_BETA: oId = "TBETA"; break;
            case THOMP_NORM: oId = "TNORM"; break;
            case EGREEDY: oId = "EGREED"; break;      
          }

          switch (s) {
            case HGM: sId = "HGMT"; break;
            case UNIF: sId = "UNI"; break;
          }

          switch (r) {
            case AD_HOC: rId = "ADHOC"; break;
            case FSCORE: rId = "FSCORE"; break;
          }

          // Build the template of the output
          std::string outputName = topo + "_" + doubleToString(duration) + "_" + oId + "_" + sId + "_" + rId + "_" + doubleToString(testDuration) + "_DIFSATURATION";

          // Log
          std::cout << "Working on " << outputName << "..." << std::endl;

          // Launching nSimulations simulations
          std::vector<Simulation*> simulations(nSimulations);
          for (unsigned int i = 0; i < nSimulations; i++) {
            simulations[i] = new Simulation(o, s, r, "./scratch/nsTest/topos/" + topo + ".json", duration, testDuration, outputName + "_" + std::to_string(i) + ".tsv", times, saturatedStas);
          }

          // Wait for all simulations to finish
          for (unsigned int i = 0; i < nSimulations; i++) {
            waitpid(simulations[i]->getPID(), NULL, 0);
            delete simulations[i];
          }
          std::cout << "Simulations terminated" << std::endl;

          // Aggregate the data
          aggregateSimulationsResults("./scratch/nsTest/data/"+outputName, "./scratch/nsTest/data/"+outputName+"_",
                                      nSimulations, duration / testDuration);
          std::cout << "Aggregation terminated" << std::endl << std::endl;
        }
      }
    }
  }
}