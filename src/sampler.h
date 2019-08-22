// parSMURFng
// 2019 - Alessandro Petrini - AnacletoLAB - Universita' degli Studi di Milano
#pragma once
#include <iostream>
#include <string>
#include <vector>
#include <thread>
#include "ANN.h"

#include "parSMURFUtils.h"

class Sampler {
public:
	Sampler(size_t n, size_t m, size_t tot, size_t numOrigPos, uint32_t k, uint32_t fp);
	~Sampler() {};
	void overSample(std::vector<double> &localData);

private:
	void getSample(const size_t numSamp, std::vector<double> &localData, double * const sample);
	void setSample(const size_t numCol, std::vector<double> &localData, const double * const sample);
	void oversampleInThread(size_t th, size_t numOfThreads, std::vector<double> &localData, ANNkd_tree * const kdTree, uint32_t local_k, double randMax);

	const size_t		n;
	const size_t		m;
	const size_t		tot;
	const size_t		numOrigPos;

	const uint32_t 		k;
	const uint32_t		fp;
};
