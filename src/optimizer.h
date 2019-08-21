// parSMURFng
// 2019 - Alessandro Petrini - AnacletoLAB - Universita' degli Studi di Milano
#pragma once
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <numeric>
#include <thread>
#include <mutex>
#include <mpi.h>

#include "parSMURFUtils.h"
#include "organizer.h"
#include "MegaCache.h"
#include "curves.h"
#include "hyperSMURF_core.h"
#include "easylogging++.h"

class Optimizer {
public:
	Optimizer(int rank, int worldSize, MegaCache * const cache, CommonParams commonParams,
			std::vector<GridParams> &gridParams, Organizer &organ);
	~Optimizer() {};
	void runOpt();

	size_t								bestModelIdx = 0;
	std::vector<std::vector<double>>	auprcs;
	std::vector<std::vector<double>>	aurocs;

private:
	GridParams getNextParams(bool &endReached);
	void partProcess(int rank, int worldSize, size_t thrNum, MegaCache * const cache, Organizer &organ,
			CommonParams &commonParams, GridParams gridParams, std::vector<size_t> &partsForThisRank,
			uint8_t currentFold, size_t internalDiv, std::mutex * p_accumulLock, std::mutex * p_partVectLock, std::vector<double> &preds);
	void evaluatePartialCurves(const std::vector<double> &preds, const std::vector<size_t> &posTest,
			const std::vector<size_t> &negTest, double * const auroc, double * const auprc);
	GridParams helpMeObiOneKenobiYouAreMyOnlyHope(bool &endReached);
	void clearPending(GridParams currentParams);

	int							rank;
	int							worldSize;
	MegaCache * const			cache;
	CommonParams				commonParams;
	std::vector<GridParams>		&gridParams;
	std::vector<GridParams>		internalGridParams;
	Organizer					&organ;
	size_t						paramsIdx = 0;
};
