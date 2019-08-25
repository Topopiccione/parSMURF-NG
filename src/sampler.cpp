// parSMURFng
// 2019 - Alessandro Petrini - AnacletoLAB - Universita' degli Studi di Milano
#include "sampler.h"

Sampler::Sampler(size_t n, size_t m, size_t tot, size_t numOrigPos, uint32_t k, uint32_t fp) :
		n{n}, m{m}, tot{tot}, numOrigPos{numOrigPos}, k{k}, fp{fp} {
}

void Sampler::overSample(std::vector<double> &localData) {
	uint32_t localk = (numOrigPos >= k) ? k : numOrigPos;
	double randMax = static_cast<double>( RAND_MAX );
	ANNpointArray dataPts = annAllocPts( numOrigPos, m + 1 );	// Data points

	// load data into dataPts
	for (size_t i = 0;  i < numOrigPos; i++) {
		getSample( i, localData, dataPts[i] );
	}

	ANNkd_tree*			kdTree = nullptr;
	kdTree = new ANNkd_tree(dataPts, numOrigPos, m + 1);

	// Timer ttt;
	// ttt.startTime();

	size_t numSampleThrd = 1;
	std::vector<std::thread> sampleThreadVect(numSampleThrd);
	for (size_t i = 0; i < numSampleThrd; i++) {
		sampleThreadVect[i] = std::thread(&Sampler::oversampleInThread, this, i, numSampleThrd, std::ref(localData), kdTree, localk, randMax);
	}
	for (size_t i = 0; i < numSampleThrd; i++)
		sampleThreadVect[i].join();

	// ttt.endTime();
	// std::cout << "Sample time: " << ttt.duration() << std::endl;

	delete kdTree;
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

void Sampler::oversampleInThread(size_t th, size_t numOfThreads, std::vector<double> &localData, ANNkd_tree * const kdTree, uint32_t localk, double randMax) {
	uint32_t 				ptIndex;
	uint32_t 				idx;
	double					alpha;
	double					oneMinusAlpha;
	std::vector<double>		tempPt(m + 1);
	ANNpoint				queryPt = annAllocPt( m + 1 );			// query point
	ANNidxArray				nnIdx	= new ANNidx[localk];			// near neighbor indices
	ANNdistArray			dists	= new ANNdist[localk];			// near neighbor distances

	for (size_t i = th; i < numOrigPos; i += numOfThreads) {
		getSample(i, localData, queryPt);
		kdTree->annkSearch(queryPt, localk, nnIdx, dists, 0);
		for (size_t j = 0; j < fp; j++) {
			idx = rand() % (localk - 1);
			getSample(nnIdx[idx], localData, tempPt.data());
			alpha = rand() / randMax;
			oneMinusAlpha = 1.0 - alpha;
			for (uint32_t l = 0; l < m; l++) {
				tempPt[l] = tempPt[l] * alpha + queryPt[l] * oneMinusAlpha;
			}
			tempPt[m] = 1.0;
			ptIndex = numOrigPos + i * fp + j;
			setSample( ptIndex, localData, tempPt.data() );
		}
	}
	annDeallocPt( queryPt );
	delete[] dists;
	delete[] nnIdx;
}


// inline
// void Sampler::getSample(const size_t m, const size_t tot, const size_t numSamp, std::vector<double> &localData, double * const sample) {
// 	for (size_t i = 0; i < m + 1; i++)
// 		sample[i] = localData[numSamp + tot * i];
// }
//
// inline
// void Sampler::setSample(const size_t m, const size_t tot, const size_t numCol, std::vector<double> &localData, const double * const sample) {
// 	for (uint32_t i = 0; i < m + 1; i++)
// 		localData[numCol + tot * i] = sample[i];
// }
//
// void Sampler::oversampleThrd(size_t th, size_t numOfThreads) {
// 	// Created and used inside the thread
// 	uint32_t		ptIndex;
// 	uint32_t		idx;
// 	double			alpha;
// 	double			oneMinusAlpha;
// 	double 			randMax = static_cast<double>( RAND_MAX );
//
// 	// Newly assigned at each generation (to be set outside)
// 	std::vector<double> &localData;
// 	uint32_t		*	localk;
// 	ANNkd_tree		*	kdTree;
// 	Sampler			*	samp;
//
// 	// Ciclo while di attesa
// 	// Set pointers and go
//
//
// 	std::vector<double>		tempPt(samp->m + 1);
// 	ANNpoint				queryPt = annAllocPt(samp->m + 1);		// query point
// 	ANNidxArray				nnIdx	= new ANNidx[localk];			// near neighbor indices
// 	ANNdistArray			dists	= new ANNdist[localk];			// near neighbor distances
//
// 	for (size_t i = th; i < samp->numOrigPos; i += numOfThreads) {
// 		getSample(i, samp->localData, queryPt);
// 		samp->kdTree->annkSearch(queryPt, samp->localk, nnIdx, dists, 0);
// 		for (size_t j = 0; j < samp->fp; j++) {
// 			idx = rand() % (samp->localk - 1);
// 			Sampler::getSample(samp->m, samp->tot, nnIdx[idx], samp->localData, tempPt.data());
// 			alpha = rand() / randMax;
// 			oneMinusAlpha = 1.0 - alpha;
// 			for (uint32_t l = 0; l < m; l++) {
// 				tempPt[l] = tempPt[l] * alpha + queryPt[l] * oneMinusAlpha;
// 			}
// 			tempPt[m] = 1.0;
// 			ptIndex = samp->numOrigPos + i * samp->fp + j;
// 			Sampler::setSample(samp->m, samp->tot, ptIndex, samp->localData, tempPt.data());
// 		}
// 	}
// 	annDeallocPt( queryPt );
// 	delete[] dists;
// 	delete[] nnIdx;
// }
