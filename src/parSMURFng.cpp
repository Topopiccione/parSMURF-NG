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
#include "organizer.h"
#include "runner.h"
#include "easylogging++.h"

#include "hyperSMURF_core.h"

INITIALIZE_EASYLOGGINGPP

int main(int argc, char ** argv){
	START_EASYLOGGINGPP(argc, argv);
	int	rank = 0;
	int worldSize = 1;

	double cacheTime, organTime, runnerTime;
	Timer ttt;

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
	if ((rank == 0) & commandLine.printCurrentConfig)
		commandLine.printConfig(0,0);

	// Megacache and organizer init
	if (rank == 0)
		ttt.startTime();
	MegaCache mc(rank, worldSize, commonParams);
	if (rank == 0) {
		ttt.endTime();
		cacheTime = ttt.duration();
		ttt.startTime();
	}
	Organizer organ(rank, &mc, commonParams);
	if (rank == 0) {
		ttt.endTime();
		organTime = ttt.duration();
		ttt.startTime();
	}

	// Create an instance of the runner and launch the run
	Runner runner(rank, worldSize, &mc, organ, commonParams, gridParams);
	runner.go();
	if (rank == 0) {
		ttt.endTime();
		runnerTime = ttt.duration();
	}
	if ((commonParams.wmode == MODE_CV) | (commonParams.wmode == MODE_PREDICT))
		runner.savePredictions();

	if ((rank == 0) && (commonParams.timeFilename != "")) {
		// Append computation time to log file, if specified by option 'exec':'timeFile'
		std::ofstream timeFile( commonParams.timeFilename.c_str(), std::ios::out | std::ios::app );
		timeFile << "#Working procs: " << worldSize
				<< " - #ensThreads: " << commonParams.nThr
				<< " - #rfThreads: " << commonParams.rfThr
				<< " - #folds: " << (uint32_t) commonParams.nFolds
				<< " - #parts: " << gridParams[0].nParts
				<< " - #fp: " << gridParams[0].fp
				<< " - #ratio: " << gridParams[0].ratio
				<< " - #nTrees: " << gridParams[0].nTrees
				<< " - #mtry: " << gridParams[0].mtry
				<< " - Cache time: " << cacheTime
				<< " - Organizer time: " << organTime
				<< " - Computation time: " << runnerTime
				<< std::endl;
		timeFile.close();
	}

	LOG(TRACE) << TXT_BIYLW << "rank: " << rank << " arrived at final barrier." << TXT_NORML;
	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Finalize();

	return 0;
}
