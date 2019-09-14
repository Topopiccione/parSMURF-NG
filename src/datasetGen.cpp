#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <fstream>

/*
######################################################################
# Function to generate synthetic imbalanced data
# A variable number of minority and majority class examples are generated. All the features of the majority
# class are distributed according to a gausian distributin with mean=0 and sd=1. Of the overall
# n.features n.inf.features of the minority class are distributed according to a gaussian centered in 1
# with standard deviation sd.
# Input:
# n.pos: number of positive (minority clsss) examples (def. 100)
# n.neg: number of negative (majority class) examples  (def. 2000)
# n.feaures: total number of features (def. 10)
# n.inf.features: number of informative features (def. 2)
# sd: standard deviation of the informative features (def.1)
# seed: intialization seed for the random number generator. If 0 (def) current clock time is used.
# Output:
# A list with two elements:
# data: the matrix of the synthetic data having pos+n.neg rows and n.features columns
# labels: a factor with the labels of he examples: 1 for minority and 0 for majority class.
# construction of a synthetic unbalanced data set
imbalanced.data.generator <- function(n.pos=100, n.neg=2000, n.features=10, n.inf.features=2, sd=1, seed=0) {
  if (seed!=0)
     set.seed(seed);
  class0 <- matrix(rnorm(n.neg*n.features, mean=0, sd=1), nrow=n.neg);
  class1 <-matrix(rnorm(n.pos*n.inf.features, mean=1, sd=sd), nrow=n.pos);
  classr1<-matrix(rnorm(n.pos*(n.features-n.inf.features), mean=0, sd=1), nrow=n.pos);
  class1 <- cbind(class1,classr1);
  data <- rbind(class1,class0);
  labels<-factor(c(rep(1,n.pos),rep(0,n.neg)), levels=c("1","0"));
  return (list(data=data, labels=labels));
}

*/

int main_old(int argc, char ** argv) {
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


int main(int argc, char ** argv) {
	if (argc < 9) {
		std::cout << "usage:" << std::endl;
		std::cout << "datasetGen <nOfPosSamples> <nOfNegSamples> <nOfFeatures> <nOfInformativeFeatures> <stdev> <dataFilename> <labelFilename> <seed>" << std::endl;
		exit(0);
	}

	size_t nPos = (size_t) std::atoi(argv[1]);
	size_t nNeg = (size_t) std::atoi(argv[2]);
	size_t nFeats = (size_t) std::atoi(argv[3]);
	size_t nInfFeats = (size_t) std::atoi(argv[4]);
	float stdDev = (float) std::atof(argv[5]);
	std::string dataFilename(argv[6]);
	std::string labelFilename(argv[7]);
	uint32_t seed = (uint32_t) std::atoi(argv[8]);

	std::default_random_engine gen( seed );
	std::normal_distribution<> disxNegAndNotInfPos(0, 1);
	std::normal_distribution<> disxInfPos(1, stdDev);

	// Create the dataset
	std::ofstream dataFile(dataFilename.c_str(), std::ios::out);

	// Start with the positives
	for (size_t i = 0; i < nPos; i++) {
		for (size_t j = 0; j < nInfFeats; j++)
			dataFile << disxInfPos(gen) << " ";
			// dataFile << "p" << " ";
		for (size_t j = 0; j < (nFeats - nInfFeats); j++)
			dataFile << disxNegAndNotInfPos(gen) << " ";
			// dataFile << "q" << " ";
		dataFile << std::endl;
	}
	// End with the negatives
	for (size_t i = 0; i < nNeg; i++) {
		for (size_t j = 0; j < nFeats; j++)
			dataFile << disxNegAndNotInfPos(gen) << " ";
			// dataFile << "n" << " ";
		dataFile << std::endl;
	}
	dataFile.close();

	// Create the label file
	std::ofstream labelFile( labelFilename.c_str(), std::ios::out );
	for (size_t i = 0; i < nPos; i++)
		labelFile << "1" << std::endl;
	for (size_t i = 0; i < nNeg; i++)
		labelFile << "0" << std::endl;
	labelFile.close();

	return 0;
}
