// parSMURFng
// 2019 - Alessandro Petrini - AnacletoLAB - Universita' degli Studi di Milano
#pragma once
#include <iostream>
#include <vector>
#include <algorithm>
#include <mpi.h>

#include "parSMURFUtils.h"
#include "easylogging++.h"

class Folds {
public:
	Folds() {};
	Folds(int rank, std::string foldFilename, uint8_t &nFolds, size_t &nRead, std::vector<uint8_t> &labels,  const bool * const labelsImported);
	~Folds() {};

	size_t								n;
	uint8_t								nFolds;
	std::vector<std::vector<size_t>>	posIdx;
	std::vector<std::vector<size_t>>	negIdx;

private:
	void readFoldsFromFile(const std::string foldFilename, size_t &n, uint8_t &nFolds, std::vector<uint8_t> &dstVect);
	int									rank;
};
