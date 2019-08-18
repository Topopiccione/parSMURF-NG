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
#include <mpi.h>

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

// From: https://stackoverflow.com/questions/1562074/how-do-i-show-the-value-of-a-define-at-compile-time
// #define VALUE_TO_STRING(x) #x
// #define VALUE(x) VALUE_TO_STRING(x)
// #define VAR_NAME_VALUE(var) #var "="  VALUE(var)
//
// #pragma message(VAR_NAME_VALUE(SIZE_MAX))
// #pragma message(VAR_NAME_VALUE(USHRT_MAX))
// #pragma message(VAR_NAME_VALUE(UINT_MAX))
// #pragma message(VAR_NAME_VALUE(ULONG_MAX))
// #pragma message(VAR_NAME_VALUE(ULLONG_MAX))


//// MPI define for size_t. NOT PORTABLE. ONLY WORKS ON 64-bit Linux
// 64-bit arch
#define MPI_SIZE_T_ MPI_UNSIGNED_LONG_LONG
// 32-bit arch
// #define MPI_SIZE_T_ MPI_UNSIGNED
// 16-bit arch
// #define MPI_SIZE_T_ MPI_UNSIGNED_SHORT
// 8-bit arch
// #define MPI_SIZE_T_ MPI_UNSIGNED_CHAR


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
	OPT_NO			= 8,		// Disable internal CV
	OPT_GRID_CV		= 16,		// Enable internal CV with grid search for optimal parameters
	OPT_AUTOGP_CV	= 32,		// Enable internal CV with Gaussian Process search for optimal parameters
	OPT_GRID_HO		= 64,
	OPT_AUTOGP_HO	= 128
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
	size_t		nn;
	size_t		mm;
	uint8_t		nFolds;
	// uint32_t	nFolds;
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
	size_t		cacheSize;
	bool	 	rfVerbose;
	bool		verboseMPI;
	bool		noMtSender;
	bool		customCV;
	bool		foldsRandomlyGenerated;
	float		hoProportion;
	uint32_t	minFold;
	uint32_t	maxFold;
};

// Various utility functions
std::vector<std::string> generateRandomName( const int n );
std::vector<std::string> generateNames(const size_t n);
// void saveToFile( const std::vector<double> &preds, const std::vector<uint8_t> &folds, std::string outFilename );
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

template <typename T>
inline void printVect( std::vector<T> vv ) {
	std::for_each(vv.begin(), vv.end(), [&](T val) {std::cout << val << " "; });
	std::cout << std::endl;
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
