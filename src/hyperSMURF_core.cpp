// parSMURFng
// 2019 - Alessandro Petrini - AnacletoLAB - Universita' degli Studi di Milano
#include "hyperSMURF_core.h"

hyperSMURFcore::hyperSMURFcore(CommonParams commonParams, GridParams gridParams, MegaCache * const cache)
		: commonParams{commonParams}, gridParams{gridParams}, cache{cache} {

	n				= commonParams.nn;
	m				= commonParams.mm;
	seed			= commonParams.seed;
	verboseLevel	= commonParams.verboseLevel;
	rfThr			= commonParams.rfThr;
	wmode			= commonParams.wmode;
	rfVerbose		= commonParams.rfVerbose;
	forestDirname	= commonParams.forestDirname;

	nPart			= gridParams.nParts;
	numTrees		= gridParams.nTrees;
	fp				= gridParams.fp;
	ratio			= gridParams.ratio;
	k				= gridParams.k;
	mtry			= gridParams.mtry;

	rfTrain			= nullptr;
	rfTest			= nullptr;
}

void hyperSMURFcore::train(std::vector<size_t> &posIdxIn, std::vector<size_t> &negIdxIn) {
	posIdx = posIdxIn;
	negIdx = negIdxIn;
	// Evaluate data matrix size
	size_t totPos, totNeg, tot;
	{
		size_t posToBeGenerated = (fp + 1) * posIdx.size();
		size_t negToBeGenerated = posToBeGenerated * ratio;
		negToBeGenerated = (negToBeGenerated > negIdx.size()) ? negIdx.size() : negToBeGenerated;
		totPos = posToBeGenerated;
		totNeg = negToBeGenerated;
		tot    = totPos + totNeg;
	}

	// Assemble data matrix
	// In parallel: copy positive and negative samples
	localData = std::vector<double>(tot * (m + 1));
	copySamplesInLocalData(posIdx.size(), posIdx, 0, tot, localData);
	copySamplesInLocalData(totNeg, negIdx, totPos, tot, localData);

	// Oversample
	Sampler samp(m, n, tot, posIdx.size(), k, fp);
	samp.overSample(localData);

	// Forest train
	uint32_t seedCustom = seed;
	std::vector<std::string> nomi = generateNames( m + 1 );
	nomi[m] = "Labels";
	// BEWARE! DataDouble applies a std::move to localData, whose status remains undefined afterwards!
	// This may be critical if we are going to reuse localData
	std::unique_ptr<Data> input_data( new DataDouble( localData, nomi, tot, m + 1 ) );
	rfTrain = new rfRanger( m, false, std::move(input_data), numTrees, mtry, rfThr, seedCustom );
	rfTrain->train( false );
}

hyperSMURFcore::~hyperSMURFcore() {
	if (rfTrain != nullptr)
		delete rfTrain;
	if (rfTest != nullptr)
		delete rfTest;
}

inline
void hyperSMURFcore::copySamplesInLocalData(const size_t howMany, const std::vector<size_t> &idx, const size_t startIdx, const size_t tot, std::vector<double> &localData) {
	std::vector<double> tempSample(m + 1);
	for (size_t i = 0; i < howMany; i++) {
		cache->getSample(idx[i], tempSample);
		for (size_t j = 0; j < tempSample.size(); j++)
			localData[i + startIdx + j*tot] = tempSample[j];
	}
}
