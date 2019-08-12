// parSMURFng
// 2019 - Alessandro Petrini - AnacletoLAB - Universita' degli Studi di Milano
#include "MegaCache.h"

MegaCache::MegaCache(const int rank, const int worldSize, size_t cacheSize, std::string dataFileName, std::string labelFileName, std::string foldFileName) :
		rank{rank}, worldSize{worldSize}, cacheSize{cacheSize}, dataFilename{dataFileName}, labelFilename{labelFileName}, foldFilename{foldFileName},
		cacheMode{FULLCACHEMODE}, labelsImported{false}, foldsImported{false}, featuresDetected{false}, cacheReady{false} {

	// Label and folds import are managed through STL I/O functions
	// Data access is done by MPI I/O primitives
	std::thread t1( &MegaCache::detectNumberOfFeatures, this );
	std::thread t2( &MegaCache::loadLabels, this, std::ref(labels), &n, &nPos );
	std::thread t3;
	size_t tempVal;
	if (!foldFilename.empty())
		t3 = std::thread( &MegaCache::loadFolds, this, std::ref(folds), &tempVal, &nFolds );

	t1.join();
	t2.join();
	if (!foldFilename.empty())
		t3.join();

	if (!foldFilename.empty() & (tempVal != n))
		std::cout << TXT_BIRED << "WARNING: size mismatch between label and fold file!!!" << TXT_NORML << std::endl;

	size_t datasize = n * m * sizeof(double);
	std::cout << TXT_BIYLW << "Size of dataset: " << datasize << " bytes. ";
	if (datasize <= cacheSize) {
		cacheMode = FULLCACHEMODE;
		std::cout << "Enabling full cache mode." << TXT_NORML << std::endl;
		data = std::vector<double>(n * m);
		dataIdx = std::vector<size_t>(n);
		dataIdxInv = std::vector<size_t>(n);
		size_t tIdx = 0;
		std::for_each(dataIdx.begin(), dataIdx.end(), [tIdx](size_t &val) mutable {val = tIdx++;});
		std::for_each(dataIdxInv.begin(), dataIdxInv.end(), [tIdx](size_t &val) mutable {val = tIdx++;});
	} else {
		cacheMode = PARTCACHEMODE;
		std::cout << "Enabling partial cache mode." << TXT_NORML << std::endl;
		size_t tempNumElem = cacheSize / sizeof(double);
		tempNumElem /= m;
		data = std::vector<double>(tempNumElem * m);
		dataIdx = std::vector<size_t>(tempNumElem);
		dataIdxInv = std::vector<size_t>(tempNumElem);
	}
	labels = std::vector<uint8_t>(n);
	folds = std::vector<uint8_t>(n);
	// This is going to be moved to an external file, if it becomes too expensive to be kept in ram.
	// Eventually, we should think about compressing data.
	dataFileIdx = std::vector<size_t>(n);

	preloadAndPrepareData();

	cacheReady = true;
}

MegaCache::~MegaCache() {}

// Open datafile and detect the number of features (m).
// datafile must be in space separated headerless format.
// Each line is a sample, each column a feature
void MegaCache::detectNumberOfFeatures() {
	std::ifstream dataFile( dataFilename.c_str(), std::ios::in );
	if (!dataFile)
		throw std::runtime_error( TXT_BIRED + std::string("Error opening data file.") + TXT_NORML );

	// 1) detecting the number of columns
	if (rank == 0)
		std::cout << TXT_BIBLU << "Detecting the number of features from data..." << TXT_NORML << std::endl;
	// Get the length of the first line
	char c;
	size_t con = 0;
	while (dataFile.get(c)) {
		con++;
		if (c == '\n')
			break;
	}
	// Allocate a buffer and read the first line in its entirety
	char * buffer = new char[con];				checkPtr<char>( buffer, __FILE__, __LINE__ );
	dataFile.seekg (0, dataFile.beg);
	dataFile.getline(buffer, con);
	// split the string according to the standard delimiters of a csv or tsv file (space, tab, comma)
	std::vector<std::string> splittedBuffer = split_str( buffer, " ,\t" );
	std::cout << TXT_BIGRN << splittedBuffer.size() << " features detected from data file." << TXT_NORML << std::endl;
	m = splittedBuffer.size();
	dataFile.close();
	delete[] buffer;

	featuresDetected = true;
}

void MegaCache::loadLabels(std::vector<uint8_t> &dstVect, size_t * valsRead, size_t * nPos) {
	size_t con = 0;
	uint32_t inData;
	dstVect.clear();

	std::ifstream labelFile( labelFilename.c_str(), std::ios::in );
	if (!labelFile)
		throw std::runtime_error( TXT_BIRED + std::string("Error opening label file.") + TXT_NORML );

	std::cout << TXT_BIBLU << "Reading label file..." << TXT_NORML << std::endl;
	while (labelFile >> inData) {
		dstVect.push_back( (uint8_t)inData );
		con++;
	}
	std::cout << TXT_BIGRN << con << " labels read" << TXT_NORML << std::endl;
	*valsRead = con;
	labelFile.close();

	// Count positives and fill posIdx
	*nPos = (size_t) std::count( dstVect.begin(), dstVect.end(), 1 );
	std::cout << TXT_BIGRN << *nPos << " positives" << TXT_NORML << std::endl;
	posIdx = std::vector<size_t>(*nPos);
	con = 0;
	for (size_t i = 0; i < dstVect.size(); i++) {
		if (dstVect[i] > 0)
			posIdx[con++] = i;
	}

	labelsImported = true;
}

void MegaCache::loadFolds(std::vector<uint8_t> &dstVect, size_t * valsRead, uint8_t * nFolds) {
	size_t con = 0;
	uint8_t inData;
	dstVect.clear();

	std::ifstream foldFile( foldFilename.c_str(), std::ios::in );
	if (!foldFile)
		throw std::runtime_error( TXT_BIRED + std::string("Error opening fold file.") + TXT_NORML );

	std::cout << TXT_BIBLU << "Reading fold file..." << TXT_NORML << std::endl;
	*nFolds = 0;
	while (foldFile >> inData) {
		dstVect.push_back( inData );
		con++;
		if (dstVect[con - 1] > *nFolds) *nFolds = dstVect[con - 1];
	}
	std::cout << TXT_BIGRN << con << " values read." << TXT_NORML << std::endl;
	(*nFolds)++;
	std::cout << TXT_BIGRN << "Total number of folds: " << *nFolds << TXT_NORML << std::endl;
	foldFile.close();

	foldsImported = true;
}

void MegaCache::preloadAndPrepareData() {
	// When in full cache mode, import as in parSMURFn. Almost...
	// Each process has to read in the complete file => use MPI_FILE_READ_ALL
	// (collective with individual file pointers)
	if (cacheMode == FULLCACHEMODE) {
		MPI_Status	fstatus;
		size_t bufSize = 2 * 1024;		// 2 Kb per proc buffer, for testing purpose
		size_t dataRead = 0;
		size_t elementsImported = 0;
		uint8_t * buf = new uint8_t[bufSize];

		// Temporary buffer for data conversion and reminder storage
		size_t tempBufIdx = 0;
		char * tempBuf = new char[256];
		std::memset(tempBuf, '\0', 256);

		// TODO: experiment with MPI_Info
		MPI_File_open(MPI_COMM_WORLD, dataFilename.c_str(), MPI_MODE_RDONLY, MPI_INFO_NULL, &dataFile_Mpih);
		MPI_Offset filesize;
		MPI_File_get_size(dataFile_Mpih, &filesize);
		//MPI_File_set_view(dataFile_Mpih, 0, MPI_UNSIGNED_CHAR, MPI_UNSIGNED_CHAR, "native", MPI_INFO_NULL);
		std::cout << "Rank: " << rank << " - " << filesize << std::endl;
		while (dataRead < filesize) {
			std::memset(buf, ' ', bufSize);
			MPI_File_read(dataFile_Mpih, buf, bufSize, MPI_UNSIGNED_CHAR, &fstatus);
			MPI_Barrier(MPI_COMM_WORLD);
			//std::cout << TXT_BIYLW << "rank: " << rank << " - " << TXT_NORML << buf << std::endl;
			// char a;
			// std::cin >> a;
			processBuffer(buf, bufSize, tempBuf, &tempBufIdx, &elementsImported);
			dataRead += (bufSize);
			//std::cout << TXT_BIYLW << "Buffer empty" << TXT_NORML << std::endl;
		}

		std::cout << TXT_BIYLW << elementsImported << " elements imported " << TXT_NORML << std::endl;

		delete[] tempBuf;
		delete[] buf;
		MPI_File_close(&dataFile_Mpih);
	}
}

void MegaCache::processBuffer(uint8_t * const buf, const size_t bufSize, char * const tempBuf, size_t * const tempBufIdx, size_t * const elementsImported) {
	size_t idx = 0;

	while (idx < bufSize) {
		// We have a space and it could be the end of a number or an empty space:
		// if tempBufIdx > 0, convert the buffer to a number, empty the buffer and continue
		// otherwise, continue
		if ((buf[idx] == ' ') | (buf[idx] == '\n')){
			if (*tempBufIdx > 0)
				convertData(tempBuf, tempBufIdx, elementsImported);
			idx++;
			continue;
		// If it is not a space or a new line, append the char to the temporary buffer and
		// increment the idx
		} else {
			tempBuf[*tempBufIdx] = buf[idx++];
			(*tempBufIdx)++;
		}
	}
}


void MegaCache::convertData(char * const tempBuf, size_t * const tempBufIdx, size_t * const elementsImported) {
	//if (rank == 0) std::cout << tempBuf << std::endl;
	double tempVal = strtod(tempBuf, nullptr);
	data[*elementsImported] = tempVal;
	// std::cout << data[*elementsImported] << std::endl;
	(*elementsImported)++;
	std::memset(tempBuf, '\0', *tempBufIdx);
	*tempBufIdx = 0;
}
