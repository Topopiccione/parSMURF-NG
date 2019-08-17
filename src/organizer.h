// parSMURFng
// 2019 - Alessandro Petrini - AnacletoLAB - Universita' degli Studi di Milano
#pragma once
#include <iostream>
#include <string>
#include <vector>

#include "parSMURFUtils.h"
#include "MegaCache.h"
#include "easylogging++.h"

struct OrgStruct;

struct OrgStruct {
	std::vector<size_t> posTrng;
	std::vector<size_t> negTrng;
	std::vector<size_t> posTest;
	std::vector<size_t> negTest;
	std::vector<OrgStruct> internalDivision;
};

class Organizer {
public:
	Organizer(MegaCache * const cache, int rank, CommonParams commonParams);
	~Organizer() {}

	std::vector<OrgStruct>		org;

private:
	int							rank;
};
