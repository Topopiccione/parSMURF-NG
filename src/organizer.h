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
	Organizer(int rank, MegaCache * const cache, CommonParams commonParams);
	~Organizer() {}

	std::vector<OrgStruct>		org;
	CommonParams				commonParams;

private:
	void populateHO(std::vector<size_t> &posTrng, std::vector<size_t> &negTrng, OrgStruct &out);

	int							rank;
	size_t						itemCounter = 0;
};
