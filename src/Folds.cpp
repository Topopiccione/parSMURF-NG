// parSMURFng
// 2019 - Alessandro Petrini - AnacletoLAB - Universita' degli Studi di Milano
#include "Folds.h"

Folds::Folds(int rank, std::string foldFilename, uint8_t &nnFolds, size_t &nRead,
		std::vector<uint8_t> &labels) :
		rank{rank} {
	// Folds from file. This can be made concurrently and indipendently on each rank
	if (!foldFilename.empty()) {
		LOG(TRACE) << TXT_BIBLU << "Rank: " << rank << " reading folds from file" << TXT_NORML;
		std::vector<uint8_t> tempFolds;
		readFoldsFromFile(foldFilename, nRead, nnFolds, tempFolds);

		n = nRead;
		nFolds = nnFolds;
		posIdx = std::vector<std::vector<size_t>>(nFolds);
		negIdx = std::vector<std::vector<size_t>>(nFolds);

		// ...then start doing our own business
		size_t tempIdx = 0;
		std::for_each(tempFolds.begin(), tempFolds.end(), [&](uint8_t val) {
			if (labels[tempIdx] > 0)
				posIdx[val].push_back(tempIdx);
			else
				negIdx[val].push_back(tempIdx);
			tempIdx++;
		});
		LOG(INFO) << TXT_BIGRN << (uint32_t) nFolds << " folds detected from file" << TXT_NORML;
	// Folds randomly generated. Only rank 0 generate the random division, then broadcast
	// to the other ranks
	} else {
		LOG(TRACE) << TXT_BIBLU << "Rank: " << rank << " generating random folds" << TXT_NORML;
		std::vector<size_t> tempPosIdx;
		std::vector<size_t> tempNegIdx;

		nFolds = nnFolds;
		n = nRead = labels.size();
		posIdx = std::vector<std::vector<size_t>>(nFolds);
		negIdx = std::vector<std::vector<size_t>>(nFolds);

		for (size_t i = 0; i < labels.size(); i++)
			(labels[i] > 0) ? tempPosIdx.push_back(i) : tempNegIdx.push_back(i);

		// Only rank 0 shuffles, then broadcast the results.
		// This will maintain the fold division consistent across ranks.
		if (rank == 0) {
			// Shuffling
			std::random_shuffle(tempPosIdx.begin(), tempPosIdx.end());
			std::random_shuffle(tempNegIdx.begin(), tempNegIdx.end());
		}

		MPI_Bcast(tempPosIdx.data(), tempPosIdx.size(), MPI_SIZE_T_, 0, MPI_COMM_WORLD);
		MPI_Bcast(tempNegIdx.data(), tempNegIdx.size(), MPI_SIZE_T_, 0, MPI_COMM_WORLD);
		MPI_Barrier(MPI_COMM_WORLD);

		// Filling posIdx and negIdx
		for (size_t i = 0; i < tempPosIdx.size(); i++)
			posIdx[i % nFolds].push_back(tempPosIdx[i]);
		size_t tempPosSize = tempPosIdx.size();
		for (size_t i = 0; i < tempNegIdx.size(); i++)
			negIdx[(i + tempPosSize) % nFolds].push_back(tempNegIdx[i]);

		LOG(INFO) << TXT_BIGRN << (uint32_t) nFolds << " folds generated" << TXT_NORML;
	}
}


void Folds::readFoldsFromFile(const std::string foldFilename, size_t &n, uint8_t &nFolds, std::vector<uint8_t> &dstVect) {
	uint32_t inData;
	dstVect.clear();

	std::ifstream foldFile( foldFilename.c_str(), std::ios::in );
	if (!foldFile)
		throw std::runtime_error( TXT_BIRED + std::string("Error opening fold file.") + TXT_NORML );

	LOG(TRACE) << TXT_BIBLU << "Reading fold file..." << TXT_NORML;
	nFolds = 0;
	while (foldFile >> inData) {
		dstVect.push_back( (uint8_t)inData );
		if (dstVect.back() > nFolds)
			nFolds = dstVect.back();
	}
	n = dstVect.size();
	(nFolds)++;
	LOG(INFO) << TXT_BIGRN << n << " values read from fold file." << TXT_NORML;
	LOG(INFO) << TXT_BIGRN << "Total number of folds: " << (uint32_t) nFolds << TXT_NORML;
	foldFile.close();
}
