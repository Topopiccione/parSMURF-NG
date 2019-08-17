// parSMURFng
// 2019 - Alessandro Petrini - AnacletoLAB - Universita' degli Studi di Milano
#include "organizer.h"

Organizer::Organizer(int rank, MegaCache * const cache, CommonParams commonParams) :
		rank {rank} {
	// Test mode is the easiest: all the dataset is considered as test set
	// Also, ignore the fold division and fill just the testIdx vectors with values from 0 to n-1
	// TODO: Decide what to do when loading data and labels for testing mode. If no label file is
	// specified, we should load only one of the two vectors and not run any auprc evaluation of
	// the final predictions; if label file is specified, otherwise.
	// TODO: The same applies to fold generation: in test mode, we should specify only a single
	// fold (for example fold 0) or avoiding defining a fold structure at all
	//
	// We are now assuming that the fold manager generates only one fold and divides the test set
	// according to the labelling. If no label file is present, only posIdx is loaded
	if (commonParams.wmode == MODE_PREDICT) {
		OrgStruct tempOrg;
		tempOrg.posTest = cache->foldManager.posIdx[0];
		tempOrg.negTest = cache->foldManager.negIdx[0];
		org.push_back(tempOrg);
	// In CV mode, we fill the org vector with as many OrgStruct as the number of folds.
	// We need the foldManager from cache.
	} else if (commonParams.wmode == MODE_CV) {
		for (uint8_t i = 0; i < cache->foldManager.nFolds; i++) {
			OrgStruct tempOrg;
			tempOrg.posTest = cache->foldManager.posIdx[i];
			tempOrg.negTest = cache->foldManager.negIdx[i];
			// The training set is composed by all samples not belonging to the i-th fold
			// Again, for consistency, shuffling is done only on rank 0.
			for (uint8_t j = 0; j < cache->foldManager.nFolds; j++) {
				if (i == j)
					continue;
				std::for_each(cache->foldManager.posIdx[j].begin(), cache->foldManager.posIdx[j].end(), [&](size_t val) {tempOrg.posTrng.push_back(val);});
				std::for_each(cache->foldManager.negIdx[j].begin(), cache->foldManager.negIdx[j].end(), [&](size_t val) {tempOrg.negTrng.push_back(val);});
			}
			if (rank == 0)
				std::random_shuffle(tempOrg.negTrng.begin(), tempOrg.negTrng.end());

			// If an optimization mode has been activated, we also need to fill the internalDivision vector
			if (commonParams.woptimiz != OPT_NO) {
				// TODO!
			}

			MPI_Bcast(tempOrg.negTrng.data(), tempOrg.negTrng.size(), MPI_SIZE_T_, 0, MPI_COMM_WORLD);
			MPI_Barrier(MPI_COMM_WORLD);

			org.push_back(tempOrg);
		}
	// Train mode is an hybrid between predict and cv, since again we have only one fold,
	// but the optimizer could be active
	} else if (commonParams.wmode == MODE_TRAIN) {
		OrgStruct tempOrg;
		tempOrg.posTrng = cache->foldManager.posIdx[0];
		tempOrg.negTrng = cache->foldManager.negIdx[0];

		if (rank == 0) {
			std::random_shuffle(tempOrg.negTrng.begin(), tempOrg.negTrng.end());
		}

		// If ann optimization mode has been activated, we also need to fill the internalDivision vector
		if (commonParams.woptimiz != OPT_NO) {
			// TODO!
		}

		MPI_Bcast(tempOrg.negTrng.data(), tempOrg.negTrng.size(), MPI_SIZE_T_, 0, MPI_COMM_WORLD);
		MPI_Barrier(MPI_COMM_WORLD);

		org.push_back(tempOrg);
	}
}
