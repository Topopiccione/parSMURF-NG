// parSMURFng
// 2019 - Alessandro Petrini - AnacletoLAB - Universita' degli Studi di Milano
#include "runner.h"

Runner::Runner(int rank, int worldSize, MegaCache * const cache, Organizer &organ, CommonParams commonParams, std::vector<GridParams> gridParams) :
		rank{rank}, worldSize{worldSize}, cache{cache}, organ{organ},
		commonParams{commonParams}, gridParams{gridParams} {


		}

void Runner::go() {
	// CV mode, no optimization
	if (commonParams.wmode == MODE_CV) {
		// Set the number of folds according to what has been specified in the options
		uint8_t startingFold, endingFold;
		{
			startingFold = 0;
			endingFold = commonParams.nFolds;
			if (commonParams.minFold != -1)
				startingFold = commonParams.minFold;
			if (commonParams.maxFold != -1)
				endingFold = commonParams.maxFold;
		}

		// Allocate the predictions vector
		preds = std::vector<double>(commonParams.nn, 0);

		// Cycle on every fold
		for (uint8_t currentFold = startingFold; currentFold < endingFold; currentFold++) {
			LOG(TRACE) << TXT_BIBLU << "rank: " << rank << " starting fold " << currentFold << TXT_NORML;

			// Division between train and test sets have been already performed in the Organizer.
			// PosTrng set is used in every partition, while NegTrng must be subdivided.

			// Each rank creates a vector containing the idx of assigned partitions
			std::vector<size_t> partsForThisRank;
			{
				// Evaluate the number of partitions assigned to this rank
				// This formula evenly distributes the partitions among ranks
				size_t partsAssigned = gridParams[0].nParts / worldSize + ((gridParams[0].nParts % worldSize) > rank);
				// Evaluate the idx of the first partition of this rank
				size_t tempIdx = 0;
				for (size_t i = 0; i < rank; i++)
					tempIdx += gridParams[0].nParts / worldSize + ((gridParams[0].nParts % worldSize) > i);
				size_t nextIdx = tempIdx + gridParams[0].nParts / worldSize + ((gridParams[0].nParts % worldSize) > rank);
				for (size_t i = tempIdx; i < nextIdx; i++)
					partsForThisRank.push_back(i);
				// std::cout << "rank " << rank << ": ";
				// std::for_each(partsForThisRank.begin(), partsForThisRank.end(), [&](size_t val){std::cout << val << " ";});
				// std::cout << std::endl;
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
						std::ref(commonParams), std::ref(gridParams), std::ref(partsForThisRank), currentFold,
						&p_accumulLock, &p_partVectLock, std::ref(localPreds)));
				}
				for (size_t i = 0; i < commonParams.nThr; i++)
					threadVect[i].join();
			}

			// Finally, we must gather on rank 0 all the localPred vectors and accumulate them in the output vector
			{
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
				MPI_Barrier( MPI_COMM_WORLD );
			}
		}
	}
	if (rank == 0)
		printVect(preds);
}

void Runner::partProcess(int rank, int worldSize, size_t thrNum, MegaCache * const cache, Organizer &organ,
		CommonParams &commonParams, std::vector<GridParams> &gridParams, std::vector<size_t> &partsForThisRank,
		uint8_t currentFold, std::mutex * p_accumulLock, std::mutex * p_partVectLock, std::vector<double> &localPreds) {
	// We are inside a thread that processes a partition. We iterate until partsForThisRank is empty.
	while (true) {
		size_t currentPart;
		// Each thread acquires the lock and pop a value from the vector
		p_partVectLock->lock();
			if (partsForThisRank.size() > 0) {
				currentPart = partsForThisRank.back();
				partsForThisRank.pop_back();
				std::cout << "Rank " << rank << " thread " << thrNum << " - popped " << currentPart << std::endl;
			} else {
				p_partVectLock->unlock();
				break;
			}
		p_partVectLock->unlock();

		// Now that we have a partition to work on, build a localTrngNeg
		std::vector<size_t> localTrngNeg;
		{
			// Evaluate how many negative must be taken + start and end idxs in organ.org[currentFold].negTrng
			// (see partitionMPI.cpp in the old parSMURF)
			size_t totNeg = organ.org[currentFold].negTrng.size();
			size_t nParts = gridParams[0].nParts;
			size_t negInEachPartition = ceil(totNeg / (double)(nParts));
			size_t totLocalNeg = (currentPart != (nParts - 1)) ? negInEachPartition : totNeg - (negInEachPartition * (nParts - 1));
			size_t negIdx = currentPart * negInEachPartition;
			std::for_each(organ.org[currentFold].negTrng.begin() + negIdx, organ.org[currentFold].negTrng.begin() + negIdx + totLocalNeg,
				[&](size_t val) {localTrngNeg.push_back(val);});
			// std::this_thread::sleep_for(std::chrono::milliseconds(5));
		}

		// Good news, we can build an instance of hyperSMURFcore and start the computation
		GridParams gp = gridParams[0];
		{
			hyperSMURFcore hsCore(commonParams, gp, cache, currentFold, currentPart);
			hsCore.train(organ.org[currentFold].posTrng, localTrngNeg);
			hsCore.test(organ.org[currentFold].posTest, organ.org[currentFold].negTest);

			// In hsCore.class1Prob there are the predictions for the samples in the posTest and negTest
			// Now accumulate them in the local (for the rank) output vector
			size_t testSize = organ.org[currentFold].posTest.size() + organ.org[currentFold].negTest.size();
			double divider = 1.0 / gp.nParts;
			p_accumulLock->lock();
				for (size_t i = 0; i < testSize; i++)
					localPreds[i] += (hsCore.class1Prob[i] * divider);
			p_accumulLock->unlock();
		}
	}
}
