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

	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Finalize();

	return 0;
}
