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

	if (dataFilename.substr(dataFilename.size() - 4, 4).compare(".bin") == 0)
		binaryMode = true;

	// Label and folds import are managed through STL I/O functions
	// Data access is done by MPI I/O primitives
	// std::thread t1( &MegaCache::detectNumberOfFeatures, this );
	// std::thread t2( &MegaCache::loadLabels, this, std::ref(labels), &n, &nPos );
	// t1.join();
	// t2.join();
	detectNumberOfFeatures();
	loadLabels(labels, &n, &nPos);
	generateFolds();

	if (!foldFilename.empty() & (nFromFoldGen != n))
		LOG(INFO) << TXT_BIRED << "WARNING: size mismatch between label and fold file!!!" << TXT_NORML;

	size_t datasize = n * m * sizeof(float);
	LOG(INFO) << TXT_BIYLW << "Space required for complete dataset: " << datasize << " bytes. " << TXT_NORML;
	if (datasize <= cacheSize) {
		cacheMode = FULLCACHEMODE;
		LOG(INFO) << TXT_BIYLW << "Enabling full cache mode." << TXT_NORML;
		data = std::vector<float>(n * (m + 1));
		// dataIdx = std::vector<size_t>(n);
		// dataIdxInv = std::vector<size_t>(n);
		// size_t tIdx = 0;
		// std::for_each(dataIdx.begin(), dataIdx.end(), [tIdx](size_t &val) mutable {val = tIdx++;});
		// std::for_each(dataIdxInv.begin(), dataIdxInv.end(), [tIdx](size_t &val) mutable {val = tIdx++;});
	} else {
		cacheMode = PARTCACHEMODE;
		LOG(INFO) << TXT_BIYLW << "Enabling partial cache mode." << TXT_NORML;
		size_t tempNumElem = cacheSize / sizeof(float);
		tempNumElem /= m;
		data = std::vector<float>(tempNumElem * m);
		dataIdx = std::vector<size_t>(tempNumElem);
		dataIdxInv = std::vector<size_t>(tempNumElem);
		// This is going to be moved to an external file, if it becomes too expensive to be kept in ram.
		// Eventually, we should think about compressing data.
		dataFileIdx = std::vector<size_t>(n);
	}

	preloadAndPrepareData();

	commonParams.nn = n;
	commonParams.mm = m;
	commonParams.nFolds = nFolds;
	if (commonParams.minFold == -1)
		commonParams.minFold = 0;
	if (commonParams.maxFold == -1)
		commonParams.maxFold = nFolds;

	cacheReady = true;
}

MegaCache::~MegaCache() {}

// Open datafile and detect the number of features (m).
// datafile must be in space separated headerless format.
// Each line is a sample, each column a feature
void MegaCache::detectNumberOfFeatures() {
	if (binaryMode) {
		std::ifstream dataFile( dataFilename.c_str(), std::ios::binary );
		if (!dataFile) {
			std::cout <<  TXT_BIRED + std::string("Error opening data file.") + TXT_NORML << std::endl;
			std::exit(-1);
		}

		if (rank == 0)
			LOG(TRACE) << TXT_BIBLU << "Rank " << rank << ": Detecting the number of features from data..." << TXT_NORML;

		// The number of features is stored as uint32_t in the first 4 bytes of the binary data file
		char bytes[4];
		uint32_t columns = 0;
		dataFile.read(bytes, sizeof(uint32_t));
		columns = *(reinterpret_cast<uint32_t*>(bytes));
		LOG(INFO) << TXT_BIGRN << "Rank " << rank << ": " << columns << " features detected from data file." << TXT_NORML;
		m = columns;
		dataFile.close();
	} else {
		std::ifstream dataFile( dataFilename.c_str(), std::ios::in );
		if (!dataFile) {
			std::cout <<  TXT_BIRED + std::string("Error opening data file.") + TXT_NORML << std::endl;
			std::exit(-1);
		}

		// 1) detecting the number of columns
		if (rank == 0)
			LOG(TRACE) << TXT_BIBLU << "Rank " << rank << ": Detecting the number of features from data..." << TXT_NORML;
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
		LOG(INFO) << TXT_BIGRN << "Rank " << rank << ": " << splittedBuffer.size() << " features detected from data file." << TXT_NORML;
		m = splittedBuffer.size();
		dataFile.close();
		delete[] buffer;
	}

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
	if (!labelFile) {
			std::cout <<  TXT_BIRED + std::string("Error opening label file.") + TXT_NORML << std::endl;
			std::exit(-1);
		}

	LOG(TRACE) << TXT_BIBLU << "Rank " << rank << ": reading label file..." << TXT_NORML;
	while (labelFile >> inData) {
		dstVect.push_back( (uint8_t)inData );
		con++;
	}

	LOG(INFO) << TXT_BIGRN << "Rank " << rank << ": " << con << " labels read" << TXT_NORML;
	*valsRead = con;
	labelFile.close();

	// Count positives and fill posIdx
	*nPos = (size_t) std::count( dstVect.begin(), dstVect.end(), 1 );
	LOG(INFO) << TXT_BIGRN << "Rank " << rank << ": " << *nPos << " positives" << TXT_NORML;
	posIdx = std::vector<size_t>(*nPos);
	con = 0;
	for (size_t i = 0; i < dstVect.size(); i++) {
		if (dstVect[i] > 0)
			posIdx[con++] = i;
	}

	labelsImported = true;
}

void MegaCache::generateFolds() {
	foldManager = Folds(rank, foldFilename, nFolds, nFromFoldGen, labels);
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
		size_t idxInData = 0;
		uint8_t * buf = new uint8_t[bufSize];
		size_t labelCnt = 0;

		// For percentage indicator
		size_t totEl = (m + 1) * n;
		size_t step = totEl / 10;
		size_t nextStep = step;
		uint32_t percentage = 10;

		// Temporary buffer for data conversion and reminder storage
		size_t tempBufIdx = 0;
		char * tempBuf = new char[256];
		std::memset(tempBuf, '\0', 256);

		// Defining datatypes for MPI_View
		MPI_Datatype vect_d;
		MPI_Offset offset;
		{
			std::vector<int> count_disp(worldSize);
			size_t tIdx = 0;
			std::for_each(count_disp.begin(), count_disp.end(), [&](int &val){val = bufSize / worldSize * tIdx++;});
			MPI_Type_vector(1, bufSize/worldSize, bufSize, MPI_UNSIGNED_CHAR, &vect_d);
			offset = (MPI_Offset) bufSize / worldSize;
		}
		MPI_Barrier(MPI_COMM_WORLD);

		MPI_Offset filesize;
		// TODO: experiment with MPI_Info
		MPI_File_open(MPI_COMM_SELF, dataFilename.c_str(), MPI_MODE_RDONLY, MPI_INFO_NULL, &dataFile_Mpih);
		MPI_File_get_size(dataFile_Mpih, &filesize);
		if (binaryMode) {
			MPI_File_seek(dataFile_Mpih, 4, MPI_SEEK_SET);	// Skip the first 4 bytes when in binary mode
			dataRead += 4;
		}
		MPI_File_set_view(dataFile_Mpih, offset, MPI_UNSIGNED_CHAR, vect_d, "native", MPI_INFO_NULL);
		// MPI_File_set_view(dataFile_Mpih, 0, MPI_UNSIGNED_CHAR, MPI_UNSIGNED_CHAR, "native", MPI_INFO_NULL);
		// NON USARE // MPI_File_set_view(dataFile_Mpih, rank * bufSize / worldSize, MPI_UNSIGNED_CHAR, MPI_UNSIGNED_CHAR, "native", MPI_INFO_NULL);

		if (rank == 0)
			LOG(INFO) << TXT_BIGRN << "Data file size (bytes) " << filesize << TXT_NORML;
		Timer ttt;
		ttt.startTime();
		while (dataRead < filesize) {
			std::memset(buf, ' ', bufSize);
			MPI_File_read_all(dataFile_Mpih, buf, bufSize, MPI_UNSIGNED_CHAR, &fstatus);
			if (binaryMode)
				processBinaryBuffer(buf, bufSize, &elementsImported, &idxInData, &labelCnt);
			else
				processBuffer(buf, bufSize, tempBuf, &tempBufIdx, &elementsImported, &labelCnt);
			dataRead += (bufSize);
			if (elementsImported > nextStep) {
				if (rank == 0)
					LOG(INFO) << TXT_BIYLW << percentage << "\% imported" << TXT_NORML;
				percentage += 10;
				nextStep += step;
			}
		}
		ttt.endTime();

		LOG(INFO) << TXT_BIGRN << "Rank " << rank << ": " << elementsImported << " elements imported " << TXT_NORML;
		LOG(INFO) << TXT_BIGRN << "Rank " << rank << ": MPI import time = " << ttt.duration() << TXT_NORML;

		delete[] tempBuf;
		delete[] buf;
		MPI_File_close(&dataFile_Mpih);

		// // Check: load with STL and compare what has been read by MPI
		// std::vector<float> dataStd;
		// {
		// 	float inFloat;
		// 	size_t con = 0;
		// 	size_t labelCnt = 0;
		// 	size_t labelIdx = 0;
		// 	std::ifstream dataFile( dataFilename.c_str(), std::ios::in );
		// 	if (!dataFile)
		// 		throw std::runtime_error( "Error opening matrix file." );
		//
		// 	ttt.startTime();
		// 	while (dataFile >> inFloat) {
		// 		dataStd.push_back( inFloat );
		// 		labelCnt++;
		// 		if (labelCnt == m) {
		// 			dataStd.push_back(labels[labelIdx++] > 0 ? 1 : 2);
		// 			labelCnt = 0;
		// 		}
		// 		con++;
		// 	}
		// 	ttt.endTime();
		// 	LOG(TRACE) << "rank " << rank << " - " << con << " values read from label file.";
		// 	LOG(TRACE) << TXT_BIYLW << "Rank: " << rank << ": STL import time = " << ttt.duration() << TXT_NORML;
		// 	dataFile.close();
		// 	// Now each rack compare what has been read via MPI
		// 	LOG(TRACE) << TXT_BIYLW << "rank " << rank << " - Checking... " << TXT_NORML;
		// 	for (size_t ii = 0; ii < dataStd.size(); ii++) {
		// 		if (dataStd[ii] != data[ii])
		// 			LOG(TRACE) << "rank " << rank << " - mismatch at " << ii << ": " << data[ii] << " -- " << dataStd[ii];
		// 	}
		// }
	}
}

void MegaCache::processBinaryBuffer(uint8_t * const buf, const size_t bufSize, size_t * const elementsImported,
		size_t * const idxInData, size_t * const labelCnt) {
	size_t idx = 0;
	float f;
	while ((idx < bufSize) & (*elementsImported < (m * n))) {
		f = *(reinterpret_cast<float*>(buf + idx));
		data[(*idxInData)] = f;
		(*elementsImported)++;
		(*idxInData)++;
		if (((*elementsImported) % m) == 0) {
			data[(*idxInData)] = labels[(*labelCnt)] == 1 ? 1.0 : 2.0;
			(*labelCnt)++;
			(*idxInData)++;
		}
		idx += sizeof(float);
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
	float tempVal = strtof(tempBuf, nullptr);
	data[*elementsImported] = tempVal;
	(*elementsImported)++;
	std::memset(tempBuf, '\0', *tempBufIdx);
	*tempBufIdx = 0;
}


// sample vector must have been preallocated as std::vector<float>(m+1)
void MegaCache::getSample(size_t idx, std::vector<float> &sample) {
	if (cacheMode == FULLCACHEMODE) {
		std::memcpy(sample.data(), data.data() + (idx * (m + 1)), (m + 1) * sizeof(float));
	} else {
		LOG(TRACE) << TXT_BIRED << "Partial cache not yet implemented..." << TXT_NORML;
	}
}

const std::vector<uint8_t> & MegaCache::getLabels() {
	return labels;
}
