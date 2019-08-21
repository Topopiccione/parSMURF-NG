// parSMURFng
// 2019 - Alessandro Petrini - AnacletoLAB - Universita' degli Studi di Milano
#include "runner.h"

Runner::Runner(int rank, int worldSize, MegaCache * const cache, Organizer &organ, CommonParams commonParams, std::vector<GridParams> gridParams) :
		rank{rank}, worldSize{worldSize}, cache{cache}, organ{organ},
		commonParams{commonParams}, gridParams{gridParams} {
}

void Runner::go() {
	size_t bestParamsIdx;
	if (commonParams.woptimiz != OPT_NO) {
		bestParamsIdx = runOptimizer();
		MPI_Barrier(MPI_COMM_WORLD);
	} else {
		bestParamsIdx = 0;
	}

	// Set the number of folds according to what has been specified in the options
	uint8_t startingFold, endingFold;
	{
		startingFold = 0;
		endingFold = commonParams.nFolds;
		if (commonParams.minFold != -1)
			startingFold = commonParams.minFold;
		if (commonParams.maxFold != -1)
			endingFold = commonParams.maxFold;
		if ((commonParams.wmode == MODE_TRAIN) | (commonParams.wmode == MODE_PREDICT)){
			LOG(INFO) << TXT_BIYLW << "rank " << rank << ": ignoring fold specifications in TRAIN and PREDICT modes" << TXT_NORML;
			startingFold = 0;
			endingFold = 1;
		}
	}

	// Allocate the predictions vector
	if ((commonParams.wmode == MODE_CV) | (commonParams.wmode == MODE_PREDICT))
		preds = std::vector<double>(commonParams.nn, 0);

	for (uint8_t currentFold = startingFold; currentFold < endingFold; currentFold++) {
		LOG(INFO) << TXT_BIBLU << "rank: " << rank << " starting fold " << (uint32_t) currentFold << TXT_NORML;

		// Division between train and test sets have been already performed in the Organizer.
		// PosTrng set is used in every partition, while NegTrng must be subdivided.

		// Each rank creates a vector containing the idx of assigned partitions
		std::vector<size_t> partsForThisRank;
		{
			// Evaluate the number of partitions assigned to this rank
			// This formula evenly distributes the partitions among ranks
			size_t partsAssigned = gridParams[bestParamsIdx].nParts / worldSize + ((gridParams[bestParamsIdx].nParts % worldSize) > rank);
			// Evaluate the idx of the first partition of this rank
			size_t tempIdx = 0;
			for (size_t i = 0; i < rank; i++)
				tempIdx += gridParams[bestParamsIdx].nParts / worldSize + ((gridParams[bestParamsIdx].nParts % worldSize) > i);
			size_t nextIdx = tempIdx + gridParams[bestParamsIdx].nParts / worldSize + ((gridParams[bestParamsIdx].nParts % worldSize) > rank);
			for (size_t i = tempIdx; i < nextIdx; i++)
				partsForThisRank.push_back(i);
		}

		// Now each rank can launch a pool of threads for partition processing
		// localPreds contains the prediction of the test set of the current folder, locally on each rank.
		std::mutex p_accumulLock;
		std::mutex p_partVectLock;
		std::vector<double> localPreds(organ.org[currentFold].posTest.size() + organ.org[currentFold].negTest.size(), 0);
		{
			std::vector<std::thread> threadVect;
			for (size_t i = 0; i < commonParams.nThr; i++) {
				threadVect.push_back(std::thread(&Runner::partProcess, this, rank, worldSize, i, cache, std::ref(organ),
					std::ref(commonParams), gridParams[bestParamsIdx], std::ref(partsForThisRank), currentFold,
					&p_accumulLock, &p_partVectLock, std::ref(localPreds)));
			}
			for (size_t i = 0; i < commonParams.nThr; i++)
				threadVect[i].join();
		}

		// Finally, we must gather on rank 0 all the localPred vectors and accumulate them in the output vector
		if ((commonParams.wmode == MODE_CV) | (commonParams.wmode == MODE_PREDICT)) {
			size_t testSize = organ.org[currentFold].posTest.size() + organ.org[currentFold].negTest.size();
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
				for (size_t i = 0; i < organ.org[currentFold].posTest.size(); i++)
					preds[organ.org[currentFold].posTest[i]] = gatherVect[cc++];
				for (size_t i = 0; i < organ.org[currentFold].negTest.size(); i++)
					preds[organ.org[currentFold].negTest[i]] = gatherVect[cc++];
			}
		}

		// Last thing, evaulate and print the partial AUROC and AUPRC for this fold
		if ((rank == 0) & (commonParams.wmode == MODE_CV)) {
			double auroc, auprc;
			evaluatePartialCurves(preds, organ.org[currentFold].posTest, organ.org[currentFold].negTest, &auroc, &auprc);
			LOG(INFO) << TXT_BICYA << "Fold " << (uint32_t) currentFold << ": auroc = " << auroc << "  -  auprc = " << auprc << TXT_NORML;
		}
		MPI_Barrier( MPI_COMM_WORLD );
	}

	// Evaluate curves on the entire set
	if ((rank == 0) & ((commonParams.wmode == MODE_CV) | (commonParams.wmode == MODE_PREDICT)) & (cache->getLabels().size() > 0)) {
		double auroc, auprc;
		evaluateFinalCurves(preds, &auroc, &auprc);
		LOG(INFO) << TXT_BICYA << "Final scores: auroc = " << auroc << "  -  auprc = " << auprc << TXT_NORML;
	}
}

void Runner::partProcess(int rank, int worldSize, size_t thrNum, MegaCache * const cache, Organizer &organ,
		CommonParams &commonParams, GridParams gridParams, std::vector<size_t> &partsForThisRank,
		uint8_t currentFold, std::mutex * p_accumulLock, std::mutex * p_partVectLock, std::vector<double> &localPreds) {
	// We are inside a thread that processes a partition. We iterate until partsForThisRank is empty.
	while (true) {
		size_t currentPart;
		// Each thread acquires the lock and pop a value from the vector
		p_partVectLock->lock();
			if (partsForThisRank.size() > 0) {
				currentPart = partsForThisRank.back();
				partsForThisRank.pop_back();
				LOG(TRACE) << "Rank " << rank << " thread " << thrNum << " - popped " << currentPart;
			} else {
				p_partVectLock->unlock();
				break;
			}
		p_partVectLock->unlock();

		// Now that we have a partition to work on, build a localTrngNeg
		std::vector<size_t> localTrngNeg;
		if ((commonParams.wmode == MODE_CV) | (commonParams.wmode == MODE_TRAIN)) {
			// Evaluate how many negative must be taken + start and end idxs in organ.org[currentFold].negTrng
			// (see partitionMPI.cpp in the old parSMURF)
			size_t totNeg = organ.org[currentFold].negTrng.size();
			size_t nParts = gridParams.nParts;
			size_t negInEachPartition = ceil(totNeg / (double)(nParts));
			size_t totLocalNeg = (currentPart != (nParts - 1)) ? negInEachPartition : totNeg - (negInEachPartition * (nParts - 1));
			size_t negIdx = currentPart * negInEachPartition;
			std::for_each(organ.org[currentFold].negTrng.begin() + negIdx, organ.org[currentFold].negTrng.begin() + negIdx + totLocalNeg,
				[&](size_t val) {localTrngNeg.push_back(val);});
			// std::this_thread::sleep_for(std::chrono::milliseconds(5));
		}

		// Good news, we can build an instance of hyperSMURFcore and start the computation
		{
			hyperSMURFcore hsCore(commonParams, gridParams, cache, currentFold, currentPart);
			if ((commonParams.wmode == MODE_CV) | (commonParams.wmode == MODE_TRAIN))
				hsCore.train(organ.org[currentFold].posTrng, localTrngNeg);
			if (commonParams.wmode == MODE_TRAIN) {
				hsCore.saveTrainedForest(currentPart);
			}

			if ((commonParams.wmode == MODE_CV) | (commonParams.wmode == MODE_PREDICT)) {
				hsCore.test(currentPart, organ.org[currentFold].posTest, organ.org[currentFold].negTest);
				// In hsCore.class1Prob there are the predictions for the samples in the posTest and negTest
				// Now accumulate them in the local (for the rank) output vector
				size_t testSize = organ.org[currentFold].posTest.size() + organ.org[currentFold].negTest.size();
				double divider = 1.0 / gridParams.nParts;
				p_accumulLock->lock();
					for (size_t i = 0; i < testSize; i++)
						localPreds[i] += (hsCore.class1Prob[i] * divider);
				p_accumulLock->unlock();
			}
		}
	}
}

void Runner::savePredictions() {
	if (rank == 0) {
		std::ofstream outFile( commonParams.outFilename.c_str(), std::ios::out );
		std::for_each( preds.begin(), preds.end(), [&outFile]( double nnn ) { outFile << nnn << " "; } );
		outFile << std::endl;
		if (commonParams.foldsRandomlyGenerated) {
			std::vector<uint8_t> ff(commonParams.nn);
			for (uint8_t i = 0; i < organ.org.size(); i++) {
				std::for_each(organ.org[i].posTest.begin(), organ.org[i].posTest.end(), [&ff, i](size_t val) {ff[val] = i;});
				std::for_each(organ.org[i].negTest.begin(), organ.org[i].negTest.end(), [&ff, i](size_t val) {ff[val] = i;});
			}
			std::for_each( ff.begin(), ff.end(), [&outFile]( uint8_t nnn ) { outFile << (uint32_t) nnn << " "; } );
			outFile << std::endl;
		}
		outFile.close();
	}
}

void Runner::evaluatePartialCurves(const std::vector<double> &preds, const std::vector<size_t> &posTest,
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

void Runner::evaluateFinalCurves(const std::vector<double> &preds, double * const auroc, double * const auprc) {
	Curves ccc(cache->getLabels(), preds.data());
	// BUG: Do not invert evalAUROC_ok() and evalAUPRC()...
	*auroc = ccc.evalAUROC_ok();
	*auprc = ccc.evalAUPRC();
}

size_t Runner::runOptimizer() {
	// Optimization run like this:
	// 1 select a combination of hyper-parameters or the grid or by BO, Each fold has an
	//      Organizer.OrgStruct.internalDivision which represent an internal HO or an internal CV.
	//      Supposing an external 5-fold CV, and internal HO, we have one HO for each fold => 5 intenal auprc values;
	//      Supposing an external 5-fold CV, and internal 4-fold CV, we have 5*4 internal auprc values
	// 2 run a train/test session for each of the 5 internal HO, average the auprc and store it somewhere;
	//   if in internal CV, perform the train/test on every internal fold for every internalDivision and average
	//   each value. We will have 5 averaged auprc. Average it and store it somewhere
	// 3 select the next grid combination, or interrogate the BO, and repeat
	// 4 at the end of this loop, select the hyper-parameter combination that on average scored best
	// 5 do a Runner::go() with this combination for testing the combination on every test set
	Optimizer opt(rank, worldSize, cache, commonParams, gridParams, organ);
	opt.runOpt();
	return opt.bestModelIdx;
}
