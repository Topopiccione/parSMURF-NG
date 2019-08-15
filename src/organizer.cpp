// parSMURFng
// 2019 - Alessandro Petrini - AnacletoLAB - Universita' degli Studi di Milano
#include "organizer.h"

Organizer::Organizer(MegaCache * const cache, CommonParams commonParams, GridParams gridParams) {
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
	if (commonParams.wMode == MODE_PREDICT) {
		tempOrg OrgStruct;
		tempOrg.posTest = cache->foldManager.posIdx[0];
		tempOrg.negTest = cache->foldManager.negIdx[0];
	}


}
