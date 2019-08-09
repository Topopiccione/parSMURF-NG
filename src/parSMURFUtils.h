// parSMURFng
// 2019 - Alessandro Petrini - AnacletoLAB - Universita' degli Studi di Milano
#pragma once
#include <vector>
#include <string>
#include <algorithm>
#include <random>
#include <cinttypes>
#include <fstream>
#include <chrono>
#include <iostream>

// ANSI console command for text coloring
#ifdef __unix
#define TXT_BICYA "\033[96;1m"
#define TXT_BIPRP "\033[95;1m"
#define TXT_BIBLU "\033[94;1m"
#define TXT_BIYLW "\033[93;1m"
#define TXT_BIGRN "\033[92;1m"
#define TXT_BIRED "\033[91;1m"
#define TXT_NORML "\033[0m"
#else
#define TXT_BICYA ""
#define TXT_BIPRP ""
#define TXT_BIBLU ""
#define TXT_BIYLW ""
#define TXT_BIGRN ""
#define TXT_BIRED ""
#define TXT_NORML ""
#endif


// some useful labelling...
enum verbLvl {
	VERBSILENT = 0,
	VERBPROGR  = 1,
	VERBRF     = 2,
	VERBALL    = 3
};

// operation modes
enum wmode {
	MODE_CV			= 1,
	MODE_TRAIN		= 2,
	MODE_PREDICT 	= 4
};

// Optimizer modes
enum woptimizer {
	OPT_NO			= 16,		// Disable internal CV
	OPT_GRID		= 32,		// Enable internal CV with grid search for optimal parameters
	OPT_AUTOGP		= 64		// Enable internal CV with Gaussian Process search for optimal parameters
};

struct GridParams {
	uint32_t nParts;
	uint32_t fp;
	uint32_t ratio;
	uint32_t k;
	uint32_t nTrees;
	uint32_t mtry;
	double   auroc;
	double   auprc;
};

struct CommonParams {
	uint32_t	nn;
	uint32_t	mm;
	uint32_t	nFolds;
	uint32_t	seed;
	uint32_t	verboseLevel;
	std::string dataFilename;
	std::string labelFilename;
	std::string foldFilename;
	std::string	outFilename;
	std::string	timeFilename;
	std::string	forestDirname;
	std::string	cfgFilename;
	uint32_t	nThr;
	uint32_t	rfThr;
	uint8_t		wmode;
	uint8_t		woptimiz;
	bool	 	rfVerbose;
	bool		verboseMPI;
	bool		noMtSender;
	bool		customCV;
	uint32_t	minFold;
	uint32_t	maxFold;
};

// Various utility functions
std::vector<std::string> generateRandomName( const int n );
void saveToFile( const double * const cl1, const double * const cl2, const uint32_t nn,
		const std::vector<uint32_t> * const labels, std::string outFilename );
std::vector<std::string> split_str( std::string s, std::string delimiters );
void printData(const double * const xx, const uint32_t * const yy, const size_t nn, const size_t mm, const bool printLabels );
void transposeMatrix(double * const dst, const double * const src, const size_t nn, const size_t mm);
void checkLoggerConfFile();

template <typename T>
inline void checkPtr( T * pointer, const char * file, int line ) {
	if (pointer == nullptr) {
		std::cout << TXT_BIRED << "Invalid allocation in " << file << " at line " << line << ". GAME OVER, YEEEEEEEEEEEAH!..." << TXT_NORML << std::endl;
		abort();
	}
}

class Timer {
public:
	Timer();
	void startTime();
	void endTime();
	double duration();

protected:
	std::chrono::time_point<std::chrono::high_resolution_clock> start, end;
};
