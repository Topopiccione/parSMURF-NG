// parSMURFng
// 2019 - Alessandro Petrini - AnacletoLAB - Universita' degli Studi di Milano
#include "optimizer.h"

Optimizer::Optimizer(int rank, int worldSize, MegaCache * const cache, CommonParams commonParams,
		std::vector<GridParams> &gridParams, Organizer &organ) : rank{rank}, worldSize{worldSize}, cache{cache},
		commonParams{commonParams}, gridParams{gridParams}, organ{organ} {
	if ((commonParams.woptimiz == OPT_AUTOGP_CV) | (commonParams.woptimiz == OPT_AUTOGP_HO))
		gridParams.clear();
}

void Optimizer::runOpt() {
	// Allocate a vector for the scores: one vector for each fold
	aurocs = std::vector<std::vector<double>>(commonParams.nFolds);
	auprcs = std::vector<std::vector<double>>(commonParams.nFolds);
	bool endReached = false;

	// Get the next parameter combination
	while (true) {
		GridParams currentParams = getNextParams(endReached);
		if (endReached)
			break;

		std::vector<double> auprcsPerFold;
		std::vector<double> aurocsPerFold;

		std::vector<double> preds(commonParams.nn, 0);

		for (uint8_t currentFold = 0; currentFold < commonParams.nFolds; currentFold++) {
			std::vector<double> tempAuprcs;
			std::vector<double> tempAurocs;
			for (size_t internalDiv = 0; internalDiv < organ.org[currentFold].internalDivision.size(); internalDiv++) {
				if (rank == 0)
					LOG(INFO) << TXT_BIBLU << "OPT: Starting fold " << (uint32_t) currentFold << " - internal division " << internalDiv << TXT_NORML;
				// We now perform train and test on the current internal HO/CV (comments in Runner class)
				std::vector<size_t> partsForThisRank;
				{
					size_t partsAssigned = currentParams.nParts / worldSize + ((currentParams.nParts % worldSize) > rank);
					size_t tempIdx = 0;
					for (size_t i = 0; i < rank; i++)
						tempIdx += currentParams.nParts / worldSize + ((currentParams.nParts % worldSize) > i);
					size_t nextIdx = tempIdx + currentParams.nParts / worldSize + ((currentParams.nParts % worldSize) > rank);
					for (size_t i = tempIdx; i < nextIdx; i++)
						partsForThisRank.push_back(i);
				}

				std::mutex p_accumulLock;
				std::mutex p_partVectLock;
				std::vector<double> localPreds(organ.org[currentFold].internalDivision[internalDiv].posTest.size() + organ.org[currentFold].internalDivision[internalDiv].negTest.size(), 0);
				{
					std::vector<std::thread> threadVect;
					for (size_t i = 0; i < commonParams.nThr; i++) {
						threadVect.push_back(std::thread(&Optimizer::partProcess, this, rank, worldSize, i, cache, std::ref(organ),
							std::ref(commonParams), currentParams, std::ref(partsForThisRank), currentFold, internalDiv,
							&p_accumulLock, &p_partVectLock, std::ref(localPreds)));
					}
					for (size_t i = 0; i < commonParams.nThr; i++)
						threadVect[i].join();
				}

				size_t testSize = organ.org[currentFold].internalDivision[internalDiv].posTest.size() + organ.org[currentFold].internalDivision[internalDiv].negTest.size();
				std::vector<double> gatherVect;
				if (rank == 0) {
					gatherVect = std::vector<double>(testSize * worldSize, 0);
				}
				MPI_Gather( localPreds.data(), testSize, MPI_DOUBLE, gatherVect.data(), testSize, MPI_DOUBLE, 0, MPI_COMM_WORLD );
				MPI_Barrier( MPI_COMM_WORLD );
				// Accumulate in the first part of the vector
				if (rank == 0) {
					for (size_t i = 0; i < testSize; i++) {
						for (size_t j = 1; j < (size_t)worldSize; j++) {
							gatherVect[i] += gatherVect[i + j * testSize];
						}
					}
					// And copy the results to the output vector
					size_t cc = 0;
					for (size_t i = 0; i < organ.org[currentFold].internalDivision[internalDiv].posTest.size(); i++)
						preds[organ.org[currentFold].internalDivision[internalDiv].posTest[i]] = gatherVect[cc++];
					for (size_t i = 0; i < organ.org[currentFold].internalDivision[internalDiv].negTest.size(); i++)
						preds[organ.org[currentFold].internalDivision[internalDiv].negTest[i]] = gatherVect[cc++];
				}

				{
					double auroc, auprc;
					evaluatePartialCurves(preds, organ.org[currentFold].internalDivision[internalDiv].posTest, organ.org[currentFold].internalDivision[internalDiv].negTest, &auroc, &auprc);
					tempAuprcs.push_back(auprc);
					tempAurocs.push_back(auroc);
					if (rank == 0)
						LOG(INFO) << TXT_BICYA << "OPT: Ext CV Fold " << (uint32_t) currentFold << " - Int CV/HO " << internalDiv << ": auroc = " << auroc << "  -  auprc = " << auprc << TXT_NORML;
				}
				MPI_Barrier( MPI_COMM_WORLD );
			}

			// We exhausted all the internal divisions for this fold. Evaluate the average and push to auprcs[currentFold]
			auprcsPerFold.push_back(std::accumulate(tempAuprcs.begin(), tempAuprcs.end(), 0.0) / (double)tempAuprcs.size());
			aurocsPerFold.push_back(std::accumulate(tempAurocs.begin(), tempAurocs.end(), 0.0) / (double)tempAurocs.size());
			if (rank == 0)
				LOG(INFO) << TXT_BICYA << "OPT: Ext CV Fold " << (uint32_t) currentFold << " - average on all internal CV/HO: auroc = " << aurocsPerFold.back() << "  -  auprc = " << auprcsPerFold.back() << TXT_NORML;
		}

		// Copy the auprcs and aurocs in the main vectors
		{
			for (size_t ii = 0; ii < commonParams.nFolds; ii++) {
				aurocs[ii].push_back(aurocsPerFold[ii]);
				auprcs[ii].push_back(auprcsPerFold[ii]);
			}
		}

		// Save the average in the gridParams struct
		currentParams.auroc = std::accumulate(aurocsPerFold.begin(), aurocsPerFold.end(), 0.0) / (double)aurocsPerFold.size();
		currentParams.auprc = std::accumulate(auprcsPerFold.begin(), auprcsPerFold.end(), 0.0) / (double)auprcsPerFold.size();
		if (rank == 0) {
			LOG(INFO) << TXT_BICYA << "OPT: Current parameters: " <<
				" - nParts: " << currentParams.nParts <<
				" - fp: " << currentParams.fp <<
				" - ratio: " << currentParams.ratio <<
				" - k: " << currentParams.k <<
				" - numTrees: " << currentParams.nTrees <<
				" - mtry: " << currentParams.mtry << TXT_NORML;
			LOG(INFO) << TXT_BICYA << "OPT: Params opt final score: auroc = " << currentParams.auroc << "  -  auprc = " << currentParams.auprc << TXT_NORML << std::endl;
		}

		internalGridParams.push_back(currentParams);
		if ((commonParams.woptimiz == OPT_AUTOGP_CV) | (commonParams.woptimiz == OPT_AUTOGP_HO)) {
			gridParams.push_back(currentParams);
			// Remove the "pending" status from the point that has been just evaluated
			clearPending(currentParams);
		}
	}

	// Time to find the best combination and save its index for future use
	{
		double maxAuprc = 0;
		for (size_t ii = 0; ii < internalGridParams.size(); ii++) {
			if (internalGridParams[ii].auprc > maxAuprc) {
				maxAuprc = internalGridParams[ii].auprc;
				bestModelIdx = ii;
			}
		}
		if (rank == 0) {
			LOG(INFO) << TXT_BICYA << "OPT: Optimal parameters: idx = " << bestModelIdx <<
				" - nParts: " << internalGridParams[bestModelIdx].nParts <<
				" - fp: " << internalGridParams[bestModelIdx].fp <<
				" - ratio: " << internalGridParams[bestModelIdx].ratio <<
				" - k: " << internalGridParams[bestModelIdx].k <<
				" - numTrees: " << internalGridParams[bestModelIdx].nTrees <<
				" - mtry: " << internalGridParams[bestModelIdx].mtry << TXT_NORML;
			LOG(INFO) << TXT_BICYA << "OPT: auroc = " << internalGridParams[bestModelIdx].auroc << "  -  auprc = " << internalGridParams[bestModelIdx].auprc << TXT_NORML;
		}
	}
}


GridParams Optimizer::getNextParams(bool &endReached) {
	// If grid search has been selected, return the next combo available
	if ((commonParams.woptimiz == OPT_GRID_CV) | (commonParams.woptimiz == OPT_GRID_HO)) {
		if (paramsIdx == gridParams.size()) {
			endReached = true;
			return gridParams[paramsIdx];
		} else
			return gridParams[paramsIdx++];
	} else {
		GridParams toBeReturned = {0,0,0,0,0,0,0,0};
		if (rank == 0)
			toBeReturned = helpMeObiOneKenobiYouAreMyOnlyHope(endReached);
		MPI_Bcast(&endReached, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);		// This is ugly
		MPI_Bcast(&toBeReturned, sizeof(toBeReturned), MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);
		MPI_Barrier(MPI_COMM_WORLD);
		LOG(TRACE) << "Rank " << rank <<
			" - nParts: " << toBeReturned.nParts << " - fp: " << toBeReturned.fp <<	" - ratio: " << toBeReturned.ratio <<
			" - k: " << toBeReturned.k << " - numTrees: " << toBeReturned.nTrees <<	" - mtry: " << toBeReturned.mtry;
		return toBeReturned;
	}
}


void Optimizer::partProcess(int rank, int worldSize, size_t thrNum, MegaCache * const cache, Organizer &organ,
		CommonParams &commonParams, GridParams gridParams, std::vector<size_t> &partsForThisRank,
		uint8_t currentFold, size_t internalDiv, std::mutex * p_accumulLock, std::mutex * p_partVectLock, std::vector<double> &localPreds) {
	// We are inside a thread that processes a partition. We iterate until partsForThisRank is empty.
	while (true) {
		size_t currentPart;
		// Each thread acquires the lock and pop a value from the vector
		p_partVectLock->lock();
			if (partsForThisRank.size() > 0) {
				currentPart = partsForThisRank.back();
				partsForThisRank.pop_back();
				LOG(TRACE) << "OPT: Rank " << rank << " thread " << thrNum << " - popped " << currentPart;
			} else {
				p_partVectLock->unlock();
				break;
			}
		p_partVectLock->unlock();

		std::vector<size_t> localTrngNeg;
		{
			size_t totNeg = organ.org[currentFold].internalDivision[internalDiv].negTrng.size();
			size_t nParts = gridParams.nParts;
			size_t negInEachPartition = ceil(totNeg / (double)(nParts));
			size_t totLocalNeg = (currentPart != (nParts - 1)) ? negInEachPartition : totNeg - (negInEachPartition * (nParts - 1));
			size_t negIdx = currentPart * negInEachPartition;
			std::for_each(organ.org[currentFold].internalDivision[internalDiv].negTrng.begin() + negIdx, organ.org[currentFold].internalDivision[internalDiv].negTrng.begin() + negIdx + totLocalNeg,
				[&](size_t val) {localTrngNeg.push_back(val);});
		}

		{
			hyperSMURFcore hsCore(commonParams, gridParams, cache, currentFold, currentPart);
			hsCore.train(organ.org[currentFold].internalDivision[internalDiv].posTrng, localTrngNeg);
			hsCore.test(currentPart, organ.org[currentFold].internalDivision[internalDiv].posTest, organ.org[currentFold].internalDivision[internalDiv].negTest);

			size_t testSize = organ.org[currentFold].internalDivision[internalDiv].posTest.size() + organ.org[currentFold].internalDivision[internalDiv].negTest.size();
			double divider = 1.0 / gridParams.nParts;
			p_accumulLock->lock();
				for (size_t i = 0; i < testSize; i++)
					localPreds[i] += (hsCore.class1Prob[i] * divider);
			p_accumulLock->unlock();
		}
	}
}

void Optimizer::evaluatePartialCurves(const std::vector<double> &preds, const std::vector<size_t> &posTest,
		const std::vector<size_t> &negTest, double * const auroc, double * const auprc) {
	std::vector<double> tempPreds(posTest.size() + negTest.size());
	std::vector<uint8_t> tempLabs(posTest.size() + negTest.size());
	// Copy predicitions in the temporary vector
	size_t idx = 0;
	std::for_each(posTest.begin(), posTest.end(), [&tempPreds, preds, &idx](size_t val){tempPreds[idx++] = preds[val];});
	std::for_each(negTest.begin(), negTest.end(), [&tempPreds, preds, &idx](size_t val){tempPreds[idx++] = preds[val];});
	std::fill(tempLabs.begin(), tempLabs.begin() + posTest.size(), 1);
	std::fill(tempLabs.begin() + posTest.size(), tempLabs.end(), 0);

	Curves ccc(tempLabs, tempPreds.data());
	// BUG: Do not invert evalAUROC_ok() and evalAUPRC()...
	*auroc = ccc.evalAUROC_ok();
	*auprc = ccc.evalAUPRC();
}

GridParams Optimizer::helpMeObiOneKenobiYouAreMyOnlyHope(bool &endReached) {
	// Interrogate the oracle
	std::string commandLine = std::string("python3 optimizer.py " + commonParams.cfgFilename);
	int retVal = std::system( commandLine.c_str() );

	// Open the tempOpt.txt file and get a pending point to be evaluated
	std::string fileLine;
	std::ifstream tempFile( "tempOpt.txt", std::ios::out );
	GridParams tempGridParam{0, 0, 0, 0, 0, 0, 0, 0};
	while (std::getline( tempFile, fileLine )) {
		std::vector<std::string> splittedStr = split_str(fileLine, " ");
		if (splittedStr[0].compare("DONE") == 0) {
			endReached = true;
			return tempGridParam;
		}
		if (splittedStr[6].compare("P") == 0) {
			LOG(TRACE) << "Params read: " << splittedStr[0] << " " << splittedStr[1] << " " << splittedStr[2] << " "
				<< splittedStr[3] << " " << splittedStr[4] << " " << splittedStr[5] << std::endl;
			tempGridParam.nParts	= atoi( splittedStr[0].c_str() );
			tempGridParam.fp		= atoi( splittedStr[1].c_str() );
			tempGridParam.ratio		= atoi( splittedStr[2].c_str() );
			tempGridParam.k			= atoi( splittedStr[3].c_str() );
			tempGridParam.nTrees	= atoi( splittedStr[4].c_str() );
			tempGridParam.mtry		= atoi( splittedStr[5].c_str() );
			tempFile.close();
			return tempGridParam;
		}
	}
	tempFile.close();
}

void Optimizer::clearPending(GridParams currentParams) {
	std::string fileLine;
	std::ifstream tempFile( "tempOpt.txt", std::ios::out );
	std::ofstream tempOutFile( "tempOpt__.txt", std::ios::out );
	while (std::getline( tempFile, fileLine )) {
		std::vector<std::string> splittedStr = split_str(fileLine, " ");
		if (splittedStr[6].compare("P") == 0) {
			tempOutFile << std::to_string(currentParams.nParts) << " " << std::to_string(currentParams.fp) << " " <<
				std::to_string(currentParams.ratio) << " " << std::to_string(currentParams.k) << " " <<
				std::to_string(currentParams.nTrees) << " " << std::to_string(currentParams.mtry) << " " <<
				std::to_string( -(currentParams.auprc) ) << std::endl;
		} else {
			tempOutFile << fileLine << std::endl;
		}
	}
	tempFile.close();
	tempOutFile.close();
	std::rename("tempOpt__.txt", "tempOpt.txt");
}
