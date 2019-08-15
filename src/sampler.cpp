// parSMURFng
// 2019 - Alessandro Petrini - AnacletoLAB - Universita' degli Studi di Milano
#include "sampler.h"

Sampler::Sampler(size_t n, size_t m, size_t tot, size_t numOrigPos, uint32_t k, uint32_t fp) :
		n{n}, m{m}, tot{tot}, numOrigPos{numOrigPos}, k{k}, fp{fp} {
}

void Sampler::overSample(std::vector<double> &localData) {
	uint32_t localk = (numOrigPos >= k) ? k : numOrigPos;

	ANNpointArray		dataPts = annAllocPts( numOrigPos, m + 1 );	// Data points
	ANNpoint			queryPt = annAllocPt( m + 1 );			// query point
	ANNidxArray			nnIdx	= new ANNidx[localk];			// near neighbor indices
	ANNdistArray		dists	= new ANNdist[localk];			// near neighbor distances
	std::vector<double>	tempPt(m + 1);

	// load data into dataPts
	for (size_t i = 0;  i < numOrigPos; i++) {
		getSample( i, localData, dataPts[i] );
	}

	ANNkd_tree*			kdTree = nullptr;
	kdTree = new ANNkd_tree(dataPts, numOrigPos, m + 1);

	// This will be the index for accessing the localData array via the setSample function;
	uint32_t ptIndex = numOrigPos;
	uint32_t idx;
	double alpha;
	double randMax = static_cast<double>( RAND_MAX );

	for (size_t i = 0; i < numOrigPos; i++) {
		getSample( i, localData, queryPt );
		kdTree->annkSearch( queryPt, localk, nnIdx, dists, 0 );
		for (uint32_t j = 0; j < fp; j++) {
			idx = rand() % (localk - 1);
			getSample( nnIdx[idx], localData, tempPt.data() );
			for (uint32_t l = 0; l < m; l++) {			// to m and not to m+1, because...
				alpha = rand() / randMax;
				tempPt[l] = tempPt[l] * alpha + queryPt[l] * (1 - alpha);
			}
			tempPt[m] = 1.0;							// ...the last item of the array is the label: it should not be interpolated!
			setSample( ptIndex, localData, tempPt.data() );
			ptIndex++;
		}
	}

	delete kdTree;
	delete[] dists;
	delete[] nnIdx;
	annDeallocPt( queryPt );
	annDeallocPts( dataPts );
	annClose();
}

inline
void Sampler::getSample(const size_t numSamp, std::vector<double> &localData, double * const sample) {
	for (size_t i = 0; i < m + 1; i++)
		sample[i] = localData[numSamp + tot * i];
}

inline
void Sampler::setSample(const size_t numCol, std::vector<double> &localData, const double * const sample) {
	for (uint32_t i = 0; i < m + 1; i++)
		localData[numCol + tot * i] = sample[i];
}
