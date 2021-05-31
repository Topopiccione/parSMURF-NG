#include <iostream>
#include <fstream>
#include <stdint.h>
#include <string>
#include <vector>
#include <cstdlib>

std::vector<std::string> split_str(std::string s, std::string delimiters) {
	std::vector<std::string> toBeRet;
	size_t current;
	size_t next = -1;
	do {
		current = next + 1;
		next = s.find_first_of( delimiters, current );
		if (s.substr( current, next - current ) != "")
 			toBeRet.push_back( s.substr( current, next - current ) );
	} while (next != std::string::npos);
	return toBeRet;
}

uint32_t detectNumberOfFeatures(std::string dataFilename) {
	std::ifstream dataFile( dataFilename.c_str(), std::ios::in );

	// Get the length of the first line
	char c;
	size_t con = 0;
	while (dataFile.get(c)) {
		con++;
		if (c == '\n')
			break;
	}
	// Allocate a buffer and read the first line in its entirety
	char * buffer = new char[con];
	dataFile.seekg (0, dataFile.beg);
	dataFile.getline(buffer, con);
	// split the string according to the standard delimiters of a csv or tsv file (space, tab, comma)
	std::vector<std::string> splittedBuffer = split_str( buffer, " ,\t" );
	uint32_t m = splittedBuffer.size();
	dataFile.close();
	delete[] buffer;

	return m;
}


int main(int argc, char ** argv) {
	if (argc < 2) {
		std::cout << "usage:" << std::endl;
		std::cout << "data2bin <dataFile.txt>" << std::endl;
		exit(0);
	}

	std::string dataFilename(argv[1]);
	std::string outFilename = dataFilename.substr(0, dataFilename.size() - 4) + ".bin";

	uint32_t columns = detectNumberOfFeatures(dataFilename);
	std::cout << columns << " features detected" << std::endl;

	std::ifstream fin(dataFilename.c_str(), std::ios::in);
	std::ofstream fout(outFilename.c_str(), std::ios::binary);

	float inData;
	uint32_t iddd;
	char bytes[4];
	size_t cnt = 0, nlines = 0;

	// Save the number of columns as first element
	bytes[0] = (columns >> 24) & 0xFF;
	bytes[1] = (columns >> 16) & 0xFF;
	bytes[2] = (columns >>  8) & 0xFF;
	bytes[3] =  columns        & 0xFF;
	fout << bytes[3] << bytes[2] << bytes[1] << bytes[0];
	cnt++;

	// CLAMOROSAMENTE BACATO... Non gestisce i NaN
	/*
	while (fin >> inData) {
		iddd = *(reinterpret_cast<uint32_t*>(&inData));
		bytes[0] = (iddd >> 24) & 0xFF;
		bytes[1] = (iddd >> 16) & 0xFF;
		bytes[2] = (iddd >>  8) & 0xFF;
		bytes[3] =  iddd        & 0xFF;
		fout << bytes[3] << bytes[2] << bytes[1] << bytes[0];
		cnt++;
		if ((cnt % 500000) == 0)
			std::cout << cnt << std::endl;
	}*/

	while(!fin.eof()) {
		std::string line;
		std::getline(fin, line);
		nlines++;
		std::vector<std::string> splittedBuffer = split_str(line, " ,\n" );

		for(const std::string& element : splittedBuffer) {
			//std::cout << element << " ";
        	inData = std::strtof(element.c_str(), nullptr);
			//std::cout << inData << std::endl;
			iddd = *(reinterpret_cast<uint32_t*>(&inData));
			bytes[0] = (iddd >> 24) & 0xFF;
			bytes[1] = (iddd >> 16) & 0xFF;
			bytes[2] = (iddd >>  8) & 0xFF;
			bytes[3] =  iddd        & 0xFF;
			fout << bytes[3] << bytes[2] << bytes[1] << bytes[0];
			cnt++;
			if ((cnt % 500000) == 0)
				std::cout << "\r" << cnt << std::flush;
		}
	}

	std::cout << std::endl << "Lines: " << nlines << " - elements: " << cnt << std::endl;
	fin.close();
	fout.close();
	return 0;
}
