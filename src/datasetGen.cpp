#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <fstream>

int main(int argc, char ** argv) {
	if (argc < 7) {
		std::cout << "usage:" << std::endl;
		std::cout << "datasetGen <nOfSamples> <nOfFeatures> <nOfPositiveSamples> <dataFilename> <labelFilename> <seed>" << std::endl;
		exit(0);
	}

	size_t n = (size_t) std::atoi(argv[1]);
	size_t m = (size_t) std::atoi(argv[2]);
	float posProb = (float) std::atof(argv[3]);
	std::string dataFilename(argv[4]);
	std::string labelFilename(argv[5]);
	uint32_t seed = (uint32_t) std::atoi(argv[6]);

	std::default_random_engine gen( seed );
	std::normal_distribution<> disxNeg( 0, 1 );
	std::normal_distribution<> disxPos( 0, 3 );
	std::bernoulli_distribution disy( posProb );

	std::vector<uint32_t> yy(n);	// Labels
	std::for_each( yy.begin(), yy.end(), [disy, &gen]( uint32_t &nnn ) mutable { nnn = disy( gen ); } );
	std::ofstream labelFile( labelFilename.c_str(), std::ios::out );
	std::for_each( yy.begin(), yy.end(), [&labelFile]( uint32_t nnn ) { labelFile << nnn << " "; } );
	labelFile << std::endl;
	labelFile.close();

	std::vector<double> x(m);
	std::ofstream dataFile( dataFilename.c_str(), std::ios::out );

	size_t posCount = 0;

	for (uint32_t i = 0; i < n; i++) {
		std::for_each( x.begin(), x.end(), [disxNeg, &gen]( double &nnn ) mutable { nnn = disxNeg( gen ); } );
		if (yy[i] == 1) {
			posCount++;
			std::for_each( x.begin(), x.end(), [disxPos, &gen]( double &nnn ) mutable { nnn += disxPos( gen ); } );
		}

		std::for_each( x.begin(), x.end(), [&dataFile]( double nnn ) { dataFile << nnn << " "; } );
		dataFile << std::endl;
	}
	dataFile.close();

	std::cout << n << " samples of " << m << " features generated." << std::endl;
	std::cout << posCount << " positives" << std::endl;
}
