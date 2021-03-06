// parSMURFng
// 2019 - Alessandro Petrini - AnacletoLAB - Universita' degli Studi di Milano
#pragma once
#include <iostream>
#include <string>
#include <vector>

#include "parSMURFUtils.h"
#include "MegaCache.h"
#include "sampler.h"
#include "rfRanger.h"
#include "globals.h"
#include "ForestProbability.h"
#include "DataDouble.h"
#include "DataFloat.h"
#include "easylogging++.h"

class hyperSMURFcore {
public:
	hyperSMURFcore(const CommonParams commonParams, const GridParams gridParams, MegaCache * const cache, uint8_t currentFold, size_t currentPart);
	~hyperSMURFcore();

	void train(std::vector<size_t> &posIdx, std::vector<size_t> &negIdx);
	void saveTrainedForest(size_t currentPart);
	void test(size_t currentPart, std::vector<size_t> &posIdxIn, std::vector<size_t> &negIdxIn);
	void freeTestSet();

	// Public vars
	std::vector<double>				class1Prob;

private:
	void copySamplesInLocalData(const size_t howMany, const std::vector<size_t> &idx, const size_t startIdx, const size_t tot, std::vector<double> &localData);

	// Arguments
	size_t							n;
	size_t							m;
	CommonParams					commonParams;
	GridParams						gridParams;
	MegaCache * const				cache;
	uint8_t							currentFold;	// these are used only in seed computation!
	size_t							currentPart;	// ----- --- ---- ---- -- ---- ------------

	std::vector<size_t>				posIdx;
	std::vector<size_t>				negIdx;

	// Run parameters
	uint32_t						nPart;
	uint32_t						numTrees;
	uint32_t						fp;
	uint32_t						ratio;
	uint32_t						k;
	uint32_t						mtry;
	uint32_t						seed;
	uint32_t						verboseLevel;
	uint32_t						rfThr;
	uint32_t						wmode;
	uint32_t						rfVerbose;
	std::string						forestDirname;

	// Internals
	rfRanger	*					rfTrain;
	rfRanger	*					rfTest;
	std::vector<double>				localData;
	std::vector<std::string>		nomi;

};
