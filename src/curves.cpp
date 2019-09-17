// parSMURFng
// 2019 - Alessandro Petrini - AnacletoLAB - Universita' degli Studi di Milano
#include "curves.h"

Curves::Curves() : preds( nullptr ), labls(std::vector<uint8_t>(0)) {}

Curves::Curves(const std::vector<uint8_t> & labels, const double * const predictions) :
		labls(labels), preds(predictions) {

	precision	= std::vector<float>(labls.size(), 0);	// Dummy init
	recall		= std::vector<float>(labls.size(), 0);	// Dummy init
	recall2		= std::vector<float>(labls.size() + 1);
	fpr			= std::vector<float>(labls.size() + 1);
	TP			= std::vector<size_t>(labls.size());
	FP			= std::vector<size_t>(labls.size());
	totP		= (size_t) std::count(labls.begin(), labls.end(), 1);
	totN		= (size_t) std::count(labls.begin(), labls.end(), 0);

	size_t idx = 0;
	tempLabs = std::vector<uint8_t>(labls.size());
	tempPreds = std::vector<float>(labls.size());
	// -- create a copy of the labs array for fast arithmetics
	std::for_each(tempLabs.begin(), tempLabs.end(), [&](uint8_t &val) {val = (uint8_t) labls[idx++];});
	// -- create a copy of the predictions array for fast arithmetics
	idx = 0;
	std::for_each(tempPreds.begin(), tempPreds.end(), [&](float &val) {val = (float) preds[idx++];});

	// - Sort the labels by descending prediction scores
	idx = 0;
	indexes = std::vector<size_t>(labls.size());
	// -- sort the indexes based on prediction values
	std::generate(indexes.begin(), indexes.end(), [&idx]() {return idx++;});
	std::sort(indexes.begin(), indexes.end(), [&](size_t i, size_t j) {return preds[i] > preds[j];});
	// -- apply the indexes permutation to labs and preds vectors
	apply_permutation_in_place<uint8_t, float>(tempLabs, tempPreds, indexes);
}

Curves::~Curves() {}

void Curves::evalAUPRCandAUROC(double * const out) {
	out[1] = evalAUROC_ok();
	out[0] = evalAUPRC();
}

double Curves::evalAUROC_alt() {
	size_t tempTP = 0;
	size_t tempFP = 0;
	float prevPred = -100.0f;

	// - in a loop
	size_t idx = 0;
	size_t partialIdx = 0;
	while (idx < tempLabs.size()) {
		if (tempPreds[idx] != prevPred) {
			recall2[idx] = tempTP / (float) totP;
			fpr[idx] = tempFP / (float) totN;
			prevPred = tempPreds[idx];
			partialIdx++;
		}
		if (tempLabs[idx] == 1)
			tempTP++;
		else
			tempFP++;
		idx++;
	}

	recall2[partialIdx] = tempTP / (float) totP;
	fpr[partialIdx] = tempFP / (float) totN;
	recall2.resize(partialIdx + 1);
	fpr.resize(partialIdx + 1);

	// - calculate AUROC area by trapezoidal integration
	return traps_integrate<float>(fpr, recall2);
	// - return AUROC area
}

double Curves::evalAUPRC() {
	// - filter predictions and lables associated with distinct score values
	//filter_dups<float, uint8_t>( tempPreds, tempLabs );
	// - thresholds are the distinct scores
	alphas.clear();
	alphas = tempPreds;

	TP = cumulSum<size_t, uint8_t>(tempLabs);
	std::for_each(tempLabs.begin(), tempLabs.end(), [&](uint8_t &val) { val = 1 - val;});
	FP = cumulSum<size_t, uint8_t>(tempLabs);
	FP[FP.size() - 1] = totN;

	// Calculate precision and recall
	precision.clear();
	recall.clear();
	recall.push_back(0.0);
	precision.push_back(1.0);
	for (size_t i = 0; i < FP.size(); i++) {
		precision.push_back(TP[i] / (float) (TP[i] + FP[i]));
		recall.push_back(TP[i] / (float) totP);
	}

	return traps_integrate<float>(recall, precision);
}


double Curves::evalAUROC_ok() {
	std::vector<uint8_t> tempLabs2 = tempLabs;
	std::vector<float>  tempPreds2 = tempPreds;
	alphas.clear();
	alphas = tempPreds2;

	TP = cumulSum<size_t, uint8_t>(tempLabs2);
	std::for_each(tempLabs2.begin(), tempLabs2.end(), [&](uint8_t &val) { val = 1 - val;});
	FP = cumulSum<size_t, uint8_t>(tempLabs2);
	FP[FP.size() - 1] = totN;

	fpr.clear();
	recall2.clear();
	for (size_t i = 0; i < FP.size(); i++) {
		fpr.push_back(FP[i] / (float) totN);
		recall2.push_back(TP[i] / (float) totP);
	}

	return traps_integrate<float>(fpr, recall2);
}
