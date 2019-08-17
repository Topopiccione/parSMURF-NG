// parSMURFng
// 2019 - Alessandro Petrini - AnacletoLAB - Universita' degli Studi di Milano
#pragma once
#include <iostream>
#include <string>
#include <vector>
#include <thread>
#include <mpi.h>

#include "parSMURFUtils.h"
#include "easylogging++.h"
#include "MegaCache.h"
#include "organizer.h"
#include "hyperSMURF_core.h"

class Runner{
public:
	Runner(MegaCache * const cache, Organizer &organ, CommonParams commonParams, std::vector<GridParams> gridParams);
	~Runner() {};
	void go();

private:
	MegaCache * const			cache;
	Organizer 					organ;
	CommonParams 				commonParams;
	std::vector<GridParams> 	gridParams;


};
