// parSMURFng
// 2019 - Alessandro Petrini - AnacletoLAB - Universita' degli Studi di Milano
#pragma once
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <thread>
#include <cstdlib>
#include <mpi.h>

#include "parSMURFUtils.h"
#include "Folds.h"
#include "easylogging++.h"

enum cacheMode {
	FULLCACHEMODE = 0,
	PARTCACHEMODE = 1
};

class MegaCache {
public:
	MegaCache(const int rank, const int worldSize, CommonParams &commonParams);
	~MegaCache();

	void getSample(size_t idx, std::vector<float> &sample);
	void getSamples(std::vector<size_t>, std::vector<float> &samples);
	const std::vector<uint8_t> & getLabels();

	int						rank;
	int						worldSize;
	size_t					cacheSize;
	size_t					m;				// number of features
	size_t					n;				// number of examples
	size_t					nFromFoldGen;
	uint8_t					nFolds;			// number of folds
	uint8_t					cacheMode;
	bool					labelsImported;
	bool					foldsImported;
	bool					featuresDetected;
	bool					cacheReady;

	Folds					foldManager;

private:
	void preloadAndPrepareData();
	void loadLabels(std::vector<uint8_t> &dstVect, size_t * valsRead, size_t * nPos);
	// void loadFolds(std::vector<uint8_t> &dstVect, size_t * valsRead, uint8_t * nFolds);
	void detectNumberOfFeatures();
	void processBuffer(uint8_t * const buf, const size_t bufSize, char * const tempBuf, size_t * const tempBufIdx, size_t * const elementsImported, size_t * const labelCnt);
	void processBinaryBuffer(uint8_t * const buf, const size_t bufSize, size_t * const elementsImported, size_t * const idxInData, size_t * const labelCnt);
	void convertData(char * const tempBuf, size_t * const tempBufIdx, size_t * const elementsImported);
	void generateFolds();

	std::string				dataFilename;
	std::string				labelFilename;
	std::string				foldFilename;

	CommonParams			commonParams;

	std::vector<float>		data;			// Main data array
	std::vector<uint8_t>	labels;			// Main labels array
	std::vector<uint8_t>	folds;
	std::vector<size_t>		dataIdx;		// list of idx for data and label arrays
	std::vector<size_t>		dataIdxInv;		// inverted list of idx for data and label arrays
	std::vector<size_t>		dataFileIdx;	// list of idx of lines in dataFile

	size_t					nPos;
	std::vector<size_t>		posIdx;
	size_t					currentIdx;

	MPI_File				dataFile_Mpih;

	bool					binaryMode = false;

};
