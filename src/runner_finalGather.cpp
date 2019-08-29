// parSMURFng
// 2019 - Alessandro Petrini - AnacletoLAB - Universita' degli Studi di Milano
#include "runner_finalGather.h"

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
	updateStartEndFold(startingFold, endingFold);

	// Creating the communicators
	int subRank = rank;
	int worldSubsize = worldSize;
	MPI_Comm subComm;
	MPI_Comm finalGatherComm;
	int finalGatherRank;
	// int finalRankWorldSize == commonParams.nFolds
	uint32_t parallFoldMode;
	std::vector<uint32_t> subMasterProcs(commonParams.nFolds);		// Only for PARALLELFOLDS_FULL
	std::vector<uint32_t> foldAssignedToRank;						// Only for PARALLELFOLDS_SPLITTED
	subCommCreate(startingFold, endingFold, subRank, worldSubsize, subComm, finalGatherRank, finalGatherComm, subMasterProcs, foldAssignedToRank, parallFoldMode);

	// Allocate the predictions vectors
	if ((parallFoldMode == PARALLELFOLDS_SPLITTED) & ((commonParams.wmode == MODE_CV) | (commonParams.wmode == MODE_PREDICT)))
		preds = std::vector<double>(commonParams.nn, 0);
	if ((parallFoldMode == PARALLELFOLDS_FULL) & (subRank == 0) & ((commonParams.wmode == MODE_CV) | (commonParams.wmode == MODE_PREDICT)))
		preds = std::vector<double>(commonParams.nn, 0);
	// preds = std::vector<double>(commonParams.nn, 0);
	std::vector<double> localPreds;

	for (uint8_t currentFold = startingFold; currentFold < endingFold; currentFold++) {
		LOG(INFO) << TXT_BIBLU << "rank: " << rank << " (subRank: " << subRank << ") starting fold " << (uint32_t) currentFold << TXT_NORML;

		// Division between train and test sets have been already performed in the Organizer.
		// PosTrng set is used in every partition, while NegTrng must be subdivided.

		// Each rank creates a vector containing the idx of assigned partitions
		std::vector<size_t> partsForThisRank;
		{
			// Evaluate the number of partitions assigned to this rank
			// This formula evenly distributes the partitions among ranks
			size_t partsAssigned = gridParams[bestParamsIdx].nParts / worldSubsize + ((gridParams[bestParamsIdx].nParts % worldSubsize) > subRank);
			// Evaluate the idx of the first partition of this rank
			size_t tempIdx = 0;
			for (size_t i = 0; i < subRank; i++)
				tempIdx += gridParams[bestParamsIdx].nParts / worldSubsize + ((gridParams[bestParamsIdx].nParts % worldSubsize) > i);
			size_t nextIdx = tempIdx + gridParams[bestParamsIdx].nParts / worldSubsize + ((gridParams[bestParamsIdx].nParts % worldSubsize) > subRank);
			for (size_t i = tempIdx; i < nextIdx; i++)
				partsForThisRank.push_back(i);
		}

		// Now each rank can launch a pool of threads for partition processing
		// localPreds contains the prediction of the test set of the current folder, locally on each rank.
		std::mutex p_accumulLock;
		std::mutex p_partVectLock;
		localPreds = std::vector<double>(organ.org[currentFold].posTest.size() + organ.org[currentFold].negTest.size(), 0);
		LOG(TRACE) << TXT_BIYLW << "rank: " << rank << " (subRank: " << subRank << ") starting threads" << TXT_NORML;
		{
			std::vector<std::thread> threadVect;
			for (size_t i = 0; i < commonParams.nThr; i++) {
				threadVect.push_back(std::thread(&Runner::partProcess, this, rank, subRank, worldSubsize, i, cache, std::ref(organ),
					std::ref(commonParams), gridParams[bestParamsIdx], std::ref(partsForThisRank), currentFold,
					&p_accumulLock, &p_partVectLock, std::ref(localPreds)));
			}
			for (size_t i = 0; i < commonParams.nThr; i++)
				threadVect[i].join();
		}
		LOG(TRACE) << TXT_BIYLW << "rank: " << rank << " (subRank: " << subRank << ") threads have joined" << TXT_NORML;

		// cumulate the localPredictions if more than rank is contributing to the current fold
		if ((parallFoldMode == PARALLELFOLDS_FULL) & (worldSubsize > 1) & ((commonParams.wmode == MODE_CV) | (commonParams.wmode == MODE_PREDICT))) {
			size_t testSize = organ.org[currentFold].posTest.size() + organ.org[currentFold].negTest.size();
			std::vector<double> gatherVect;
			if (subRank == 0) {
				gatherVect = std::vector<double>(testSize * worldSubsize, 0);
			}
			MPI_Gather(localPreds.data(), testSize, MPI_DOUBLE, gatherVect.data(), testSize, MPI_DOUBLE, 0, subComm);
			MPI_Barrier(subComm);
			LOG(TRACE) << TXT_BIYLW << "rank: " << rank << " (subRank: " << subRank << ") gathered on localPreds" << TXT_NORML;
			// Accumulate in the first part of the vector
			if (subRank == 0) {
				for (size_t i = 0; i < testSize; i++) {
					for (size_t j = 1; j < (size_t)worldSubsize; j++) {
						gatherVect[i] += gatherVect[i + j * testSize];
					}
				}
				std::memcpy(localPreds.data(), gatherVect.data(), testSize * sizeof(double));
			}
		}

		// Evaulate and print the partial AUROC and AUPRC for this fold
		if ((subRank == 0) & (commonParams.wmode == MODE_CV)) {
			LOG(TRACE) << TXT_BIYLW << "rank: " << rank << " (subRank: " << subRank << ") evaluating fold auprc" << TXT_NORML;
			double auroc, auprc;
			evaluatePartialCurves(localPreds, organ.org[currentFold].posTest, organ.org[currentFold].negTest, &auroc, &auprc);
			LOG(INFO) << TXT_BICYA << "Fold " << (uint32_t) currentFold << ": auroc = " << auroc << "  -  auprc = " << auprc << TXT_NORML;
		}

		// If we have less ranks than folds, copy localPreds into the local preds vector
		if (/*(parallFoldMode == PARALLELFOLDS_SPLITTED) & */(subRank == 0) & ((commonParams.wmode == MODE_CV) | (commonParams.wmode == MODE_PREDICT))) {
			size_t cc = 0;
			for (size_t i = 0; i < organ.org[currentFold].posTest.size(); i++)
				preds[organ.org[currentFold].posTest[i]] = localPreds[cc++];
			for (size_t i = 0; i < organ.org[currentFold].negTest.size(); i++)
				preds[organ.org[currentFold].negTest[i]] = localPreds[cc++];
		}

		LOG(TRACE) << TXT_BIYLW << "rank: " << rank << " (subRank: " << subRank << ") arrived at barrier" << TXT_NORML;
		MPI_Barrier( subComm );
	}

	MPI_Barrier(MPI_COMM_WORLD);

	// // Collect the localPreds from each submaster and move all the predictions to rank 0 in full parallel mode
	// // This could be done with a gather on rank 0, but it could be exepensive in terms of memeory consumption
	// if ((parallFoldMode == PARALLELFOLDS_FULL) & ((commonParams.wmode == MODE_CV) | (commonParams.wmode == MODE_PREDICT))) {
	// 	LOG(TRACE) << TXT_BIYLW << "rank: " << rank << " (subRank: " << subRank << ") final gather" << TXT_NORML;
	// 	size_t tempSize;
	// 	for (uint8_t currentFold = commonParams.minFold + 1; currentFold < commonParams.maxFold; currentFold++) {
	// 		uint8_t idxFold = currentFold - commonParams.minFold;
	// 		if (rank == subMasterProcs[idxFold]) {
	// 			LOG(TRACE) << TXT_BIYLW << "rank: " << rank << " (subRank: " << subRank << ") sending localPreds to rank 0" << TXT_NORML;
	// 			MPI_Send(localPreds.data(), localPreds.size(), MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
	// 		}
	// 		if (rank == 0) {
	// 			std::vector<double>predsFromSubRanks(organ.org[currentFold].posTest.size() + organ.org[currentFold].negTest.size());
	// 			LOG(TRACE) << TXT_BIYLW << "rank: " << rank << " (subRank: " << subRank << ") receiving localPreds from rank " << subMasterProcs[idxFold] << TXT_NORML;
	// 			MPI_Recv(predsFromSubRanks.data(), predsFromSubRanks.size(), MPI_DOUBLE, subMasterProcs[idxFold], 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	// 			LOG(TRACE) << TXT_BIYLW << "rank: " << rank << " (subRank: " << subRank << ") localPreds from rank " << subMasterProcs[idxFold] << " received" << TXT_NORML;
	// 			// And copy the results to the output vector
	// 			size_t cc = 0;
	// 			for (size_t i = 0; i < organ.org[currentFold].posTest.size(); i++)
	// 				preds[organ.org[currentFold].posTest[i]] = predsFromSubRanks[cc++];
	// 			for (size_t i = 0; i < organ.org[currentFold].negTest.size(); i++)
	// 				preds[organ.org[currentFold].negTest[i]] = predsFromSubRanks[cc++];
	// 		}
	// 		LOG(TRACE) << TXT_BIYLW << "rank: " << rank << " (subRank: " << subRank << ") waiting at barrier..." << TXT_NORML;
	// 		MPI_Barrier( MPI_COMM_WORLD );
	// 	}
	// 	LOG(TRACE) << TXT_BIYLW << "rank: " << rank << " (subRank: " << subRank << ") final gather completed" << TXT_NORML;
	//
	// 	if (rank == 0) {
	// 		size_t cc = 0;
	// 		for (size_t i = 0; i < organ.org[commonParams.minFold].posTest.size(); i++)
	// 			preds[organ.org[commonParams.minFold].posTest[i]] = localPreds[cc++];
	// 		for (size_t i = 0; i < organ.org[commonParams.minFold].negTest.size(); i++)
	// 			preds[organ.org[commonParams.minFold].negTest[i]] = localPreds[cc++];
	// 	}
	// }

	// Collect the localPreds from each submaster and move all the predictions to rank 0 in full parallel mode
	// This could be done with a gather on rank 0, but it could be exepensive in terms of memeory consumption
	if ((parallFoldMode == PARALLELFOLDS_FULL) & ((commonParams.wmode == MODE_CV) | (commonParams.wmode == MODE_PREDICT))) {
		LOG(TRACE) << TXT_BIYLW << "rank: " << rank << " (subRank: " << subRank << ") final gather" << TXT_NORML;
		std::vector<double> gatherVect;
		if (rank == 0) {
			gatherVect = std::vector<double>(preds.size() * commonParams.nFolds, 0);
		}
		MPI_Gather(preds.data(), preds.size(), MPI_DOUBLE, gatherVect.data(), preds.size(), MPI_DOUBLE, 0, finalGatherComm);
		// Accumulate in the first part of the vector
		if (rank == 0) {
			for (size_t i = 0; i < preds.size(); i++)
				for (size_t j = 1; j < (size_t)commonParams.nFolds; j++) {
					gatherVect[i] += gatherVect[i + j * preds.size()];
				}
			std::memcpy(preds.data(), gatherVect.data(), preds.size() * sizeof(double));
		}
	}

	// Collect the predictions from each rank to rank 0 in splitted parallel mode
	if ((parallFoldMode == PARALLELFOLDS_SPLITTED) & ((commonParams.wmode == MODE_CV) | (commonParams.wmode == MODE_PREDICT))) {
		for (uint32_t senderProc = 1; senderProc < worldSize; senderProc++) {
			if (rank == senderProc) {
				LOG(TRACE) << TXT_BIYLW << "rank: " << rank << " (subRank: " << subRank << ") sending predictions" << TXT_NORML;
				MPI_Send(preds.data(), preds.size(), MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
			}
			if (rank == 0) {
				std::vector<double> predsFromSubRanks(commonParams.nn, 0);
				MPI_Recv(predsFromSubRanks.data(), predsFromSubRanks.size(), MPI_DOUBLE, senderProc, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
				for (size_t i = 0; i < preds.size(); i++)
					preds[i] += predsFromSubRanks[i];
			}
			MPI_Barrier( MPI_COMM_WORLD );
		}
	}

	// Evaluate curves on the entire set
	if ((rank == 0) & ((commonParams.wmode == MODE_CV) | (commonParams.wmode == MODE_PREDICT)) & (cache->getLabels().size() > 0)) {
		double auroc, auprc;
		evaluateFinalCurves(preds, &auroc, &auprc);
		LOG(INFO) << TXT_BICYA << "Final scores: auroc = " << auroc << "  -  auprc = " << auprc << TXT_NORML;
	}

	MPI_Barrier(MPI_COMM_WORLD);
}

void Runner::partProcess(int realRank, int rank, int worldSize, size_t thrNum, MegaCache * const cache, Organizer &organ,
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
				LOG(TRACE) << "Rank " << realRank << " (subRank: " << rank << ") thread " << thrNum << " - popped " << currentPart;
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

void Runner::updateStartEndFold(uint8_t &startingFold, uint8_t &endingFold) {
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

void Runner::subCommCreate(uint8_t &startingFold, uint8_t &endingFold, int &subRank, int &worldSubsize, MPI_Comm &subComm,
		int &finalGatherRank, MPI_Comm &finalGatherComm, std::vector<uint32_t> &subMasterProcs,
		std::vector<uint32_t> &foldAssignedToRank, uint32_t &parallFoldMode) {
	int foldSpan = endingFold - startingFold;
	int ws = worldSize;
	int idx = 0;
	if (worldSize >= foldSpan) {
		int color = 0;
		std::vector<int> ppf(foldSpan);
		std::vector<uint32_t> cumul(foldSpan + 1);
		cumul[0] = 0;
		std::for_each(ppf.begin(), ppf.end(), [idx, ws, foldSpan](int &val) mutable {val = ws / foldSpan + ((ws % foldSpan) > idx++);});
		std::partial_sum(ppf.begin(), ppf.end(), cumul.begin() + 1);
		for (int i = 0; i < cumul.size(); i++) {
			if (rank < cumul[i]) {
				color = i - 1;
				break;
			}
		}
		MPI_Comm_split(MPI_COMM_WORLD, color, rank, &subComm);
		MPI_Comm_rank(subComm, &subRank);
		MPI_Comm_size(subComm, &worldSubsize);
		LOG(TRACE) << TXT_BIYLW << "Rank: " << rank << " become rank " << subRank << ". worldSubsize: " << worldSubsize << " - color: " << color << TXT_NORML;
		// Update start and ending fold
		endingFold = startingFold + (color + 1);
		startingFold += color;
		std::memcpy(subMasterProcs.data(), cumul.data(), commonParams.nFolds * sizeof(uint32_t));
		// Create the communicator for the final gather
		MPI_Comm_split(MPI_COMM_WORLD, subRank, rank, &finalGatherComm);
		MPI_Comm_rank(finalGatherComm, &finalGatherRank);
		parallFoldMode = PARALLELFOLDS_FULL;
	} else {
		MPI_Comm_split(MPI_COMM_WORLD, rank, rank, &subComm);
		MPI_Comm_rank(subComm, &subRank);
		MPI_Comm_size(subComm, &worldSubsize);
		std::vector<int> fpp(ws);
		std::vector<uint32_t> cumul(ws + 1);
		std::for_each(fpp.begin(), fpp.end(), [idx, ws, foldSpan](int &val) mutable {val = foldSpan / ws + ((foldSpan % ws) > idx++);});
		std::partial_sum(fpp.begin(), fpp.end(), cumul.begin() + 1);
		endingFold = startingFold + cumul[rank + 1];
		startingFold += cumul[rank];
		LOG(TRACE) << TXT_BIYLW << "Rank: " << rank << " (subrank " << subRank << ") assigned to fold: " << (uint32_t) startingFold << " - " << (uint32_t) endingFold << TXT_NORML;
		parallFoldMode = PARALLELFOLDS_SPLITTED;
		std::for_each(fpp.begin(), fpp.end(), [&foldAssignedToRank, idx](uint32_t val) mutable {
			for (uint32_t aa = 0; aa < val; aa++)
				foldAssignedToRank.push_back(idx);
			idx++;
		});
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
	// std::vector<double> tempPreds(posTest.size() + negTest.size());
	std::vector<uint8_t> tempLabs(posTest.size() + negTest.size());
	// Copy predicitions in the temporary vector
	size_t idx = 0;
	// std::for_each(posTest.begin(), posTest.end(), [&tempPreds, preds, &idx](size_t val){tempPreds[idx++] = preds[val];});
	// std::for_each(negTest.begin(), negTest.end(), [&tempPreds, preds, &idx](size_t val){tempPreds[idx++] = preds[val];});
	std::fill(tempLabs.begin(), tempLabs.begin() + posTest.size(), 1);
	std::fill(tempLabs.begin() + posTest.size(), tempLabs.end(), 0);

	// Curves ccc(tempLabs, tempPreds.data());
	Curves ccc(tempLabs, preds.data());
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
