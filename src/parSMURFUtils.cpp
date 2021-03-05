// parSMURFng
// 2019 - Alessandro Petrini - AnacletoLAB - Universita' degli Studi di Milano
#include "parSMURFUtils.h"

std::vector<std::string> generateRandomName(const int n) {
	const char alphanum[] =
	        "0123456789"
	        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
	        "abcdefghijklmnopqrstuvwxyz";
	std::vector<std::string> out;
	const int slen = 8;
	char stringa[slen + 1];
	stringa[slen] = 0;

	for (int i = 0; i < n; i++) {
		std::for_each( stringa, stringa + slen, [alphanum](char &c){c = alphanum[rand() % (sizeof(alphanum) - 1)];} );
		out.push_back( std::string( stringa ) );
	}
	return out;
}

std::vector<std::string> generateNames(const size_t n) {
	std::vector<std::string> out;
	for (size_t i = 0; i < n; i++)
		out.push_back(std::to_string(i));
	return out;
}

std::vector<std::string> generateOrderedNames(const size_t n) {
	std::vector<std::string> out(n);
	const int slen = 8;
	std::ostringstream ss;

	for (int i = 0; i < n; i++) {
	    ss.str(std::string());
	    int32_t expp;
	    i != 0 ? expp = (uint32_t) log10(i) : expp = 0;
		for (int o = 7; o > expp; o--)
			ss << "0";
		ss << std::to_string(i);
		out[i] = ss.str();
	}
	return out;
}

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

void printData(const double * const xx, const uint32_t * const yy, const size_t nn, const size_t mm, const bool printLabels) {
	size_t av = 0;
	for (size_t i = 0; i < nn * (mm + 1); i++) {
		if (!(i % (mm + 1)))
			std::cout << TXT_BIYLW << i / (mm + 1) << TXT_NORML << ": ";
		std::cout << xx[i] << " ";
		if (!((i+1) % (mm + 1)) & printLabels)
			std::cout << yy[av++] << std::endl;
		else if (!((i+1) % (mm + 1)) & !printLabels)
			std::cout << std::endl;
	}
}

void checkLoggerConfFile() {
	std::ifstream confFile( "logger.conf", std::ios::in );
	if (!confFile) {
		std::cout << TXT_BIYLW << "Logger configuration file not found (logger.conf). Creating one..." << std::endl;
		std::ofstream confFile( "logger.conf", std::ios::out );
		confFile << "* GLOBAL:" << std::endl;
		confFile << "    FORMAT               =  \"%datetime %msg\"" << std::endl;
		confFile << "    FILENAME             =  \"default_ParSMURFng.log\"" << std::endl;
		confFile << "    ENABLED              =  true" << std::endl;
		confFile << "    TO_FILE              =  true" << std::endl;
		confFile << "    TO_STANDARD_OUTPUT   =  true" << std::endl;
		confFile << "    SUBSECOND_PRECISION  =  6" << std::endl;
		confFile << "    PERFORMANCE_TRACKING =  true" << std::endl;
		confFile << "    MAX_LOG_FILE_SIZE    =  4194304	## 4MB" << std::endl;
		confFile << "    LOG_FLUSH_THRESHOLD  =  100 ## Flush after every 100 logs" << std::endl;
		confFile << "* DEBUG:" << std::endl;
		confFile << "    FORMAT               = \"%datetime{%d/%M} %func %msg\"" << std::endl;
		confFile << "* TRACE:" << std::endl;
		confFile << "    ENABLED              =  false" << std::endl;
		confFile << "* INFO:" << std::endl;
		confFile << "    ENABLED              =  true" << std::endl;
		confFile.close();
	} else {
		confFile.close();
	}
}


Timer::Timer() {}

double Timer::duration() {
	std::chrono::duration<double, std::ratio<1>> fp_ms = end - start;
	return fp_ms.count();
}

void Timer::startTime() {
	start = std::chrono::high_resolution_clock::now();
}

void Timer::endTime() {
	end = std::chrono::high_resolution_clock::now();
}
