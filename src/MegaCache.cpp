// parSMURFng
// 2019 - Alessandro Petrini - AnacletoLAB - Universita' degli Studi di Milano
#include "MegaCache.h"

MegaCache::MegaCache(const int rank, const int worldSize, CommonParams &commonParams) :
		rank{rank}, worldSize{worldSize}, commonParams{commonParams},
		cacheMode{FULLCACHEMODE}, labelsImported{false}, foldsImported{false}, featuresDetected{false}, cacheReady{false} {

	cacheSize		= commonParams.cacheSize;
	dataFilename	= commonParams.dataFilename;
	labelFilename	= commonParams.labelFilename;
	foldFilename	= commonParams.foldFilename;
	nFolds			= commonParams.nFolds;

	// Label and folds import are managed through STL I/O functions
	// Data access is done by MPI I/O primitives
	std::thread t1( &MegaCache::detectNumberOfFeatures, this );
	std::thread t2( &MegaCache::loadLabels, this, std::ref(labels), &n, &nPos );
	std::thread t3( &MegaCache::generateFolds, this );

	t1.join();
	t2.join();
	t3.join();


	if (!foldFilename.empty() & (nFromFoldGen != n))
		std::cout << TXT_BIRED << "WARNING: size mismatch between label and fold file!!!" << TXT_NORML << std::endl;

	size_t datasize = n * m * sizeof(double);
	std::cout << TXT_BIYLW << "Size of dataset: " << datasize << " bytes. ";
	if (datasize <= cacheSize) {
		cacheMode = FULLCACHEMODE;
		std::cout << "Enabling full cache mode." << TXT_NORML << std::endl;
		data = std::vector<double>(n * (m + 1));
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
	// This is going to be moved to an external file, if it becomes too expensive to be kept in ram.
	// Eventually, we should think about compressing data.
	dataFileIdx = std::vector<size_t>(n);

	preloadAndPrepareData();

	commonParams.nn = n;
	commonParams.mm = m;
	commonParams.nFolds = nFolds;

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
	if (rank == 0)
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

	if (labelFilename.empty() & commonParams.wmode == MODE_PREDICT) {
		labelsImported = true;
		return;
	}

	std::ifstream labelFile( labelFilename.c_str(), std::ios::in );
	if (!labelFile)
		throw std::runtime_error( TXT_BIRED + std::string("Error opening label file.") + TXT_NORML );

	if (rank == 0)
		std::cout << TXT_BIBLU << "Reading label file..." << TXT_NORML << std::endl;
	while (labelFile >> inData) {
		dstVect.push_back( (uint8_t)inData );
		con++;
	}

	std::cout << rank << std::endl;
	if (rank == 0)
		std::cout << TXT_BIGRN << con << " labels read" << TXT_NORML << std::endl;
	*valsRead = con;
	labelFile.close();

	// Count positives and fill posIdx
	*nPos = (size_t) std::count( dstVect.begin(), dstVect.end(), 1 );
	if (rank == 0)
		std::cout << TXT_BIGRN << *nPos << " positives" << TXT_NORML << std::endl;
	posIdx = std::vector<size_t>(*nPos);
	con = 0;
	for (size_t i = 0; i < dstVect.size(); i++) {
		if (dstVect[i] > 0)
			posIdx[con++] = i;
	}

	labelsImported = true;
}

void MegaCache::generateFolds() {
	foldManager = Folds(rank, foldFilename, nFolds, nFromFoldGen, labels, &labelsImported);
	foldsImported = true;
}

void MegaCache::preloadAndPrepareData() {
	// When in full cache mode, import as in parSMURFn. Almost...
	// Each process has to read in the complete file => use MPI_FILE_READ_ALL
	// (collective with individual file pointers)
	if (cacheMode == FULLCACHEMODE) {
		MPI_Status	fstatus;
		size_t bufSize = 16 * worldSize * 1024;		// 16 Kb per proc buffer
		size_t dataRead = 0;
		size_t elementsImported = 0;
		uint8_t * buf = new uint8_t[bufSize];
		size_t labelCnt = 0;

		// Temporary buffer for data conversion and reminder storage
		size_t tempBufIdx = 0;
		char * tempBuf = new char[256];
		std::memset(tempBuf, '\0', 256);

		// Defining datatypes for MPI_View
		// std::vector<int> count_proc(worldSize);
		// std::fill(count_proc.begin(), count_proc.end(), bufSize / worldSize);
		// std::vector<int> count_disp(worldSize);
		// size_t tIdx = 0;
		// std::for_each(count_disp.begin(), count_disp.end(), [&](int &val){val = bufSize / worldSize * tIdx++;});
		// MPI_Datatype vect_d;
		// MPI_Type_vector(1,bufSize/worldSize,bufSize, MPI_UNSIGNED_CHAR, &vect_d);
		// MPI_Offset offset = (MPI_Offset) count_disp[rank];
		//
		// MPI_File_set_view(dataFile_Mpih, offset, MPI_UNSIGNED_CHAR, vect_d, "native", MPI_INFO_NULL);
		MPI_File_set_view(dataFile_Mpih, rank * bufSize / worldSize, MPI_UNSIGNED_CHAR, MPI_UNSIGNED_CHAR, "native", MPI_INFO_NULL);

		// TODO: experiment with MPI_Info
		MPI_File_open(MPI_COMM_SELF, dataFilename.c_str(), MPI_MODE_RDONLY, MPI_INFO_NULL, &dataFile_Mpih);
		MPI_Offset filesize;
		MPI_File_get_size(dataFile_Mpih, &filesize);

		std::cout << "Rank: " << rank << " - " << filesize << std::endl;
		Timer ttt;
		ttt.startTime();
		while (dataRead < filesize) {
			std::memset(buf, ' ', bufSize);
			MPI_File_read_all(dataFile_Mpih, buf, bufSize, MPI_UNSIGNED_CHAR, &fstatus);
			// MPI_Barrier(MPI_COMM_WORLD);
			processBuffer(buf, bufSize, tempBuf, &tempBufIdx, &elementsImported, &labelCnt);
			dataRead += (bufSize);
		}
		ttt.endTime();

		if (rank == 0) {
			std::cout << TXT_BIYLW << elementsImported << " elements imported " << TXT_NORML << std::endl;
			std::cout << TXT_BIYLW << "Rank: " << rank << ": MPI import time = " << ttt.duration() << TXT_NORML << std::endl;
		}

		delete[] tempBuf;
		delete[] buf;
		MPI_File_close(&dataFile_Mpih);

		// Check: load with STL and compare what has been read by MPI
		std::vector<double> dataStd;
		{
			double inDouble;
			size_t con = 0;
			size_t labelCnt = 0;
			size_t labelIdx = 0;
			std::ifstream dataFile( dataFilename.c_str(), std::ios::in );
			if (!dataFile)
				throw std::runtime_error( "Error opening matrix file." );

			ttt.startTime();
			while (dataFile >> inDouble) {
				dataStd.push_back( inDouble );
				labelCnt++;
				if (labelCnt == m) {
					dataStd.push_back(labels[labelIdx++] > 0 ? 1 : 2);
					labelCnt = 0;
				}
				con++;
			}
			ttt.endTime();
			if (rank == 0) {
				std::cout << "rank " << rank << " - " << con << " values read from label file." << std::endl;
				std::cout << TXT_BIYLW << "Rank: " << rank << ": STL import time = " << ttt.duration() << TXT_NORML << std::endl;
			}
			dataFile.close();
			// Now each rack compare what has been read via MPI
			std::cout << TXT_BIYLW << "rank " << rank << " - Checking... " << TXT_NORML << std::endl;
			for (size_t ii = 0; ii < dataStd.size(); ii++) {
				if (dataStd[ii] != data[ii])
					std::cout << "rank " << rank << " - mismatch at " << ii << ": " << data[ii] << " -- " << dataStd[ii] << std::endl;
			}
		}
	}
}

void MegaCache::processBuffer(uint8_t * const buf, const size_t bufSize, char * const tempBuf,
			size_t * const tempBufIdx, size_t * const elementsImported, size_t * const labelCnt) {
	size_t idx = 0;

	while (idx < bufSize) {
		// We have a space and it could be the end of a number or an empty space:
		// if tempBufIdx > 0, convert the buffer to a number, empty the buffer and continue.
		// Otherwise, continue
		if ((buf[idx] == ' ') | (buf[idx] == '\n')){
			if (*tempBufIdx > 0)
				convertData(tempBuf, tempBufIdx, elementsImported);
			// If \n, let's put the label at the end of the line
			if (buf[idx] == '\n') {
				data[(*elementsImported)] = labels[(*labelCnt)] == 1 ? 1.0 : 2.0;
				(*elementsImported)++;
				(*labelCnt)++;
			}
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

inline void MegaCache::convertData(char * const tempBuf, size_t * const tempBufIdx, size_t * const elementsImported) {
	double tempVal = strtod(tempBuf, nullptr);
	data[*elementsImported] = tempVal;
	(*elementsImported)++;
	std::memset(tempBuf, '\0', *tempBufIdx);
	*tempBufIdx = 0;
}


// sample vector must have been preallocated as std::vector<double>(m+1)
void MegaCache::getSample(size_t idx, std::vector<double> &sample) {
	if (cacheMode == FULLCACHEMODE) {
		std::memcpy(sample.data(), data.data() + (idx * (m + 1)), (m + 1) * sizeof(double));
	} else {
		std::cout << "Not yet implemented..." << std::endl;
	}
}
