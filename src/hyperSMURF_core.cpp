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

	nomi = generateNames( m + 1 );
}

void hyperSMURFcore::train(std::vector<size_t> &posIdxIn, std::vector<size_t> &negIdxIn) {
	posIdx = posIdxIn;
	negIdx = negIdxIn;
	// Evaluate data matrix size
	size_t totPos, totNeg, tot;
	{
		totPos = (fp + 1) * posIdx.size();
		totNeg = totPos * ratio;
		totNeg = (totNeg > negIdx.size()) ? negIdx.size() : totNeg;
		tot    = totPos + totNeg;
	}

	// Assemble data matrix
	// In parallel? Copy positive and negative samples
	localData = std::vector<double>(tot * (m + 1));
	copySamplesInLocalData(posIdx.size(), posIdx, 0, tot, localData);
	copySamplesInLocalData(totNeg, negIdx, totPos, tot, localData);

	// Oversample
	Sampler samp(n, m, tot, posIdx.size(), k, fp);
	samp.overSample(localData);

	// Forest train
	uint32_t seedCustom = seed;
	nomi[m] = "Labels";
	// BEWARE! DataDouble applies a std::move to localData, whose status remains undefined afterwards!
	// This may be critical if we are going to reuse localData
	std::unique_ptr<Data> input_data( new DataDouble( localData, nomi, tot, m + 1 ) );
	rfTrain = new rfRanger( m, false, std::move(input_data), numTrees, mtry, rfThr, seedCustom );
	rfTrain->train( true );
}

void hyperSMURFcore::test(std::vector<size_t> &posIdxIn, std::vector<size_t> &negIdxIn) {
	// Assemble vector of idx
	std::vector<size_t> idxs;
	std::for_each(posIdxIn.begin(), posIdxIn.end(), [&idxs](size_t val){idxs.push_back(val);});
	std::for_each(negIdxIn.begin(), negIdxIn.end(), [&idxs](size_t val){idxs.push_back(val);});

	// Evaluate data matrix size
	size_t tot = idxs.size();

	// Assemble data matrix
	localData = std::vector<double>(tot * (m + 1));
	copySamplesInLocalData(tot, idxs, 0, tot, localData);
	// Erasing the original label of each sample in the test set
	std::for_each(localData.begin() + tot * m, localData.begin() + tot * (m + 1), [](double &val) {val = 0;});

	// Forest test
	uint32_t seedCustom = seed;
	nomi[m] = "dependent";
	// BEWARE! DataDouble applies a std::move to localData, whose status remains undefined afterwards!
	// This may be critical if we are going to reuse localData
	std::unique_ptr<Data> test_data( new DataDouble( localData, nomi, tot, m + 1 ) );
	rfTest = new rfRanger( rfTrain->forest, m, true, std::move(test_data), numTrees, mtry, rfThr, seedCustom );
	rfTest->predict( true );

	// Get predicted valued
	const std::vector<std::vector<std::vector<double>>>& predictions = rfTest->forestPred->getPredictions();
	std::cout << tot << std::endl;
	class1Prob.clear();
	std::for_each(predictions[0].begin(), predictions[0].end(), [&](std::vector<double> val) {class1Prob.push_back(val[0]);});
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
