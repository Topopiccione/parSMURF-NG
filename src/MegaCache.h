// parSMURFng
// 2019 - Alessandro Petrini - AnacletoLAB - Universita' degli Studi di Milano
#pragma once
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <thread>
#include <mpi.h>

#include "parSMURFUtils.h"
#include "easylogging++.h"

class MegaCache {
public:
	MegaCache(const int rank, std::string dataFileName, std::string labelFilename, std::string foldFilename);
	~MegaCache();

	void getSample(size_t idx, std::vector<double> &sample);
	void getSamples(std::vector<size_t>, std::vector<double> &samples);

	int						rank;
	size_t					m;				// number of features
	size_t					n;				// number of examples
	uint8_t					nFolds;			// number of folds
	bool					labelsImported;
	bool					foldsImported;
	bool					featuresDetected;
	bool					cacheReady;

private:
	void preloadAndPrepare();
	void loadLabels(std::vector<uint8_t> &dstVect, size_t * valsRead, size_t * nPos);
	void loadFolds(std::vector<uint8_t> &dstVect, size_t * valsRead, uint8_t * nFolds);
	void detectNumberOfFeatures();

	std::string				dataFilename;
	std::string				labelFilename;
	std::string				foldFilename;

	std::vector<double>		data;			// Main data array
	std::vector<uint8_t>	labels;			// Main labels array
	std::vector<uint8_t>	folds;
	std::vector<size_t>		dataIdx;		// list of idx for data and label arrays
	std::vector<size_t>		dataFileIdx;	// list of idx of lines in dataFile

	size_t					nPos;
	std::vector<size_t>		posIdx;
	size_t					currentIdx;

};
