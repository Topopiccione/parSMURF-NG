// parSMURFng
// 2019 - Alessandro Petrini - AnacletoLAB - Universita' degli Studi di Milano
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>
#include <iomanip>
#include <stdexcept>
#include <thread>
#include <cmath>
#include <mutex>
#include <mpi.h>

#include "parSMURFUtils.h"
#include "ArgHandler_new.h"
#include "MegaCache.h"
#include "easylogging++.h"

#include "hyperSMURF_core.h"

INITIALIZE_EASYLOGGINGPP

int main(int argc, char ** argv){
	START_EASYLOGGINGPP(argc, argv);
	int	rank = 0;
	int worldSize = 1;

	MPI_Init( &argc, &argv );
	MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	////EasyLogging++
	checkLoggerConfFile();
	el::Configurations conf("logger.conf");
	el::Loggers::reconfigureLogger("default", conf);
	el::Loggers::reconfigureAllLoggers(conf);

	std::vector<GridParams> gridParams;
	ArgHandle commandLine( argc, argv, gridParams );
	CommonParams commonParams = commandLine.processCommandLine( rank );

	// Megacache init
	MegaCache mc(rank, worldSize, commonParams);

	// Supertest to check if hyperSMURFcore works, using MyData1.txt and MyData1L.txt
	{
		std::vector<size_t> posTrng;
		std::vector<size_t> posTest;
		std::vector<size_t> negTrng;
		std::vector<size_t> negTest;
		size_t cc = 0;
		for (;cc < 40; cc++) posTrng.push_back(cc);
		for (;cc < 50; cc++) posTest.push_back(cc);
		for (;cc < 850; cc++) negTrng.push_back(cc);
		for (;cc < 1000; cc++) negTest.push_back(cc);

		hyperSMURFcore hsCore(commonParams, gridParams[0], &mc);
		hsCore.train(posTrng, negTrng);
		hsCore.test(posTest, negTest);

		std::for_each(hsCore.class1Prob.begin(), hsCore.class1Prob.end(),[&](double val){std::cout << val << " ";} );
		std::cout << std::endl;
	}

	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Finalize();

	return 0;
}
