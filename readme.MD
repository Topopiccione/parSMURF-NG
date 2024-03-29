# parSMURF-NG

parSMURF-NG is a High Performance Computing imbalance-aware machine learning tool for the genome-wide detection of pathogenic variants. Although being conceived for tackling bioinformatics and genomic problems, parSMURF-NG is a general purpose ML binary classifier for dealing with extremely imbalanced dataset.

---

### Table of Contents
<pre>
<a href="#Overview">Overview</a>
<a href="#Requirements">Requirements</a>
<a href="#Downloading-and-compiling">Downloading and compiling</a>
<a href="#General-architecture">General architecture</a>
<a href="#Running-parSMURF">Running parSMURF-NG</a>
	<a href="#Command line options">Command line options</a>
    <a href="#Running parSMURF-NG on a single machine">Running parSMURF-NG on a single machine</a>
    <a href="#Running parSMURF-NG on a cluster">Running parSMURF-NG on a cluster</a>
	<a href="#Running-the-Bayesian-optimizer">Running the Bayesian optimizer</a>
	<a href="#Configuration-file">Configuration file</a>
		<a href="#name">name</a>
		<a href="#exec">exec</a>
		<a href="#data">data</a>
		<a href="#simulate">simulate</a>
		<a href="#folds">folds</a>
		<a href="#params">params</a>
		<a href="#autogp_params">autogp_params</a>
<a href="#Data-format">Data Format</a>
	<a href="#Data-file">Data file format</a>
	<a href="#Label-file">Label file format</a>
	<a href="#Fold-file">Fold file format</a>
	<a href="#Output-file">Output file format</a>
<a href="#Random-dataset-generation">Random dataset generation</a>
<a href="#Examples">Examples</a>
<a href="#Licensa">License</a>
</pre>

---

### Overview
parSMURF-NG is a fast and scalable C++ implementation of the HyperSMURF algorithm - hyper-ensemble of SMOTE Undersampled Random Forests - an ensemble approach explicitly designed to deal with the huge imbalance between deleterious and neutral variants.

The algorithm is outlined in the following papers:\
A. Petrini, M. Mesiti, M. Schubach, M. Frasca, D. Danis, M. Re, G.Grossi, L. Cappelletti, T. Castrignanò, P. N. Robinson, and G. Valentini, "parSMURF, a high-performance computing tool for the genome-wide detection of pathogenic variants", GigaScience, vol. 9, 05 2020. giaa052.
https://doi.org/10.1093/gigascience/giaa052

Schubach, Matteo Re, Peter N. Robinson & Giorgio Valentini, "Imbalance-Aware Machine Learning for Predicting Rare and Common Disease-Associated Non-Coding Variants", Scientific Reports, 2017/06/07\
https://www.nature.com/articles/s41598-017-03011-5

parSMURF-NG is a more stable, faster, more scalable and up-to-date version of parSMURF https://github.com/AnacletoLAB/parSMURF . Also, usability has been improved.

parSMURF-NG has been specifically developed in the context of the project "ParBigMen: ParSMURF application to Big genomic and epigenomic data for the detection of pathogenic variants in Mendelian diseases", awarded by the Partnership of Advanced Computing in Europe.

---

### Requirements

parSMURF-NG is designed for x86-64 and Intel Xeon Phi architectures running Linux OSes.\
This software is distributed as source code.

A compilier which supports the C++11 language specification is required. It has been tested with GCC (vers. >= 5) and Intel CC (2015, 2017 and 2019).\
Code is also optimized for Intel XeonPhi architectures, and it has been successfully tested on Knights Landing family processors.This application is designed to automatically scale, from a single workstation to High Performance Computing systems.

parSMURF-NG requires an implementation of the MPI standard, as most of the I/O functions are managed through that. It has been tested with OpenMPI 1.10.3, OpenMPI 2.0, IntelMPI 2016, IntelMPI 2017 and IntelMPI 2019.
On Ubuntu, it is possible to install the OpenMPI library via apt package manager:
```
sudo apt-get install openmpi-bin openmpi-common libopenmpi-dev
```
Makefiles are generated by the cmake (vers. >= 3.0.2) utility. On Ubuntu it is possible to install this package via apt:
```
sudo apt-get install cmake
```
Bayesian Optimization is managed through the scikit-optimize Python 3 package. The best way to use this feature is by creating and configuring a Python virtual environment and installing the required Python packages there. On Ubuntu:
```
sudo apt-get install virtualenv
<move to an appropriate folder>
virtualenv parSMURFvenv -p /usr/bin/python3     # This command creates a parSMURFvenv directory
source parSMURFvenv/bin/activate		        # This command activates the virtual environment
pip install scikit-optimize    			        # The following commands install the required packages in the virtual environment
```

parSMURF uses several external libraries that are included in this repository or are automatically downloaded and compiled. In particular, the following libraries are included:
- ANN: A Library for Approximate Nearest Neighbor Searching, by David M. Mount and Sunil Arya, Version 1.1.2. The modified version is supplied in the src/ann_1.1.2 directory. This version has been adapted for multi-thread execution, since the original package available at https://www.cs.umd.edu/~mount/ANN/ is not thread safe and is not compatible with this package.
- Ranger: A Fast Implementation of Random Forests, by Marvin N. Wright, Version 0.11.2. The modified version, stripped from the R code and adapted to parSMURF-NG, is supplied in the src/ranger directory. The main codebase is located at https://github.com/imbs-hl/ranger

The following libraries are not included in this code repository, but are automatically downloaded during the compilation process:
- easylogging++: A single header C++ logging library, by Zuhd Web Services. Automatically cloned in src/easyloggingpp and compiled from https://github.com/zuhd-org/easyloggingpp
- jsoncons: A C++, header-only library for constructing JSON and JSON-like text and binary data formats, by Daniel Parker. Automatically cloned in src/jsoncons and compiled from https://github.com/danielaparker/jsoncons
- zlib: A massively spiffy yet delicately unobtrusive compression library, by Jean-loup Gailly and Mark Adler. Autmatically cloned in src/zlib and compiled from https://github.com/madler/zlib

The following library is required, but it must manually installed:
- scikit-optimize: Sequential model-based optimization in Python from https://scikit-optimize.github.io/stable/index.html

All the libraries have been modified and redistributed according to their own licenses. For each included library, a copy of the associated license is contained in its respective folder.

---

### Downloading and compiling

Download the latest version from this page or clone the git repository altogether:

	git clone https://github.com/Topopiccione/parSMURF-NG


Once the package has been downloaded, move to the main directory, create a build dir, invoke cmake and build the software ("-j n" make option enables multithread compilation over n threads):

	cd parSMURF-NG
	mkdir build
	cd build
	cmake ../src
	make -j 4

This should generate the following executables:
- parSMURFng
- data2bin
- datasetGen

---

### General architecture

parSMURF-NG is a complete rewrite of the original parSMURF software, although sharing the same core concepts and algorithm. The novelties of this package resides in a different execution model: I decided to abandon the master-worker design of parSMURF in favor of a more scalable master-less design, where each MPI process is almost independent and processes a subset of the input dataset. The only major coordination task occurs at the end of the computation, as results are gathered on MPI rank 0.

Thanks to this new design, task parallelization occurs at a finer level, always in a hierarchical way. 

The other major improvements resides in a total rewrite of all the data I/O functions, as now everything is managed by the MPI library. Data throughput when reading the dataset is notably increased.

Finally, now it is possible to specify both cross-validation OR hold-out, when designing and running an experiment.

parSMURF-NG features two subsystems for the automatic fine tuning of the learning parameters, aimed to maximize the prediction performances of the algorithm. The first strategy is by performing an exhaustive grid search: given a set of values for each hyper-parameter, the resulting set of all the possible combinations of hyper-parameters is calculated, and each combination evaluated through internal cross validation. The other strategy is by Bayesian optimization: given a range for each hyper-parameter, the Bayesian optimizer generate a sequence of possible candidates whose sequence tends to a probable global maximum. An high level of the execution is given by this pseudo-code snippet:

```
iter = 0
- while (iter < maxIter) and (error > tolerance):
-- BO generates a new possible candidate of hyper-parameters h
-- evaluation of h in a context of internal cross validation
-- submit (h, AUPRC(h)) to the BO
-- iter <- iter + 1
```

Both strategies are performed in a context of internal cross validation / hold-out.

---

### Running parSMURF-NG
parSMURF-NG is a command line executable.\
All the options are submitted to the main executable through a JSON configuration file.

#### Command line options
Only two command line options are available, since every other parameter or option is defined in the JSON configuration file.\
`--cfg <filename>` specifies the configuration file for the run\
`--help` prints a brief help screen

#### Running parSMURF-NG on a single machine
parSMURF-NG can be launched as following:
```
./parSMURFng --cfg <configFile.json>
```

#### Running parSMURF-NG on a cluster
parSMURF-NG requires MPI to be installed on the target system or in all the nodes of a cluster. It must be invoked with `mpirun` or, depending on the scheduling system installed on the cluster, with a proper mpirun wrapper.\
The `-n` option of `mpirun` also specifies how many processes have to be launched. As an example:
```
mpirun -n 5 ./parSMURFng --cfg <configFile.json>
```
launches an instance of parSMURF-NG over 5 processes.\


#### Running the Bayesian optimizer
Running the Bayesian optimizer is notably improved w.r.t. parSMURF1 and parSMURFn. If the scikit-optimize Python 3 package has been globally installed, no further operations are required and parSMURF-NG can be run as above.\
If it has been installed in a virtual environment or conda, just activate the virtual env before running parSMURF-NG.

---

#### Configuration file

parSMURF-NG uses configuration files in JSON format for setting the parameters of each run.\
Examples of configuration files are available in the cfgEx folder of the repository.

A configuration file is composed by five dictionaries:
```
{
	"name": ...,
	"exec": {...},
	"data": {...},
	"folds": {...},
	"simulate": {...},
	"params": {...}
}
```
Depending on the configuration itself, some dictionaries are not mandatory and can be left out.

##### "name"
| Name | Type | Mandatory | Default | Description |
|-|-|-|-|-|
|"name"|string|No|-|A string for labeling the name of the experiment|


##### "exec"
```
	"exec": {
		"name": string,
		"nProcs": int,
		"ensThrd": int,
		"rfThrd": int,
		"noMtSender": bool,
		"seed": int,
		"verboseLevel": int,
		"verboseMPI": bool,
		"saveTime": bool,
		"timeFile": string,
		"printCfg": bool,
		"mode": string
    },
```
Mandatory: yes\
General configuration of the run.

| Name | Type | Mandatory | Default | Description |
|-|-|-|-|-|
|"name"|string|No|-|Label used for marking the name of the executable (legacy from parSMURF1 and parSMURFn). It does not affect the computation itself, since this field is ignored by the json parser|
|"nProcs"|int|No|-|Label used for marking the number of processes for a run of parSMURFn. It does not affect the computation itself, since the total number of processes is detected at runtime by the MPI APIs.|
|"ensThrd"|int|Yes|1|Number of threads assigned to perform the partition processing|
|"oversmpThrd"|int|Yes|1|Number of threads assigned to data oversampling|
|"rfThrd"|int|Yes|1|Number of threads assigned to perform the random forest train and test|
|"noMtSender"|-|No|-|Legacy, ignored in parSMURF-NG|
|"seed"|int|No|-|Optional seed for the random number generators. If unspecified, a random seed is used|
|"verboseLevel"|int|No|0|Level of verbosity on stdout and on the logfile of the computational task. Range is 0-3|
|"verboseMPI"|bool|No|false|Verbose on stdout and logfile the calls to MPI APIs|
|"saveTime"|bool|No|false|Option for saving a report of the computation time of the run|
|"timeFile"|string|Yes, if "saveTime" is true|-|File name of the execution time report|
|"cacheSize"|string|Yes|-|This parameter specifies the cache size for data storage (1)|
|"printCfg"|bool|No|false|Option for printing a detailed description of the run before it starts|
|"mode"|string|Yes|"cv"|Execution mode (2)|
|"optimizer"|string|Yes|"no"|Execution mode (3)|

(1): Reserved for a future extension of the software to limit the memory for storing the dataset and allow parSMURF-NG to dinamically load the required data from disk. As this feature is currently not present, cache size must be equal or larger than the size of the input dataset (excluding the label and fold files). Units are specified by literals: example\
"cacheSize": "4.2G"\
"cacheSize": "1024M"\
"cacheSize": "12343K"

(2): Allowed strings are:\
"cv": Dataset is splitted in folds, and the classifier is evaluated in a process of k-fold cross validation. The run returns a set of predicted values for the whole dataset (default).\
"train": The whole dataset is treated as training set. The run returns a folder of trained models for later usage.\
"test": The whole dataset is treated as test set. It is mandatory to submit a directory of trained models to perform the evaluation. The run returns a set of predicted value for the whole dataset.\
Note that the autotuning of the learning parameters is available only for "cv" mode

(3): Allowed strings are:\
"no": external cross-validation only (default)\
"grid_cv": automatic tuning of the learning parameters by grid search in the internal cross validation loop\
"autogp_cv": automatic tuning of the learning parameters by Bayesian optimization (Gaussian process) in the internal cross validation loop\
"grid_ho": automatic tuning of the learning parameters by grid search in the internal hold-out\
"autogp_ho": automatic tuning of the learning parameters by Bayesian optimization (Gaussian process) in the internal hold-out

##### "data"
```
"data": {
	"dataFile": string
	"foldFile": string
	"labelFile": string
	"outFile": string
	"forestDir": string
}
```
Mandatory: yes\
This field contains all the required information for accessing data from and to the system.

| Name | Type | Mandatory | Default | Description |
|-|-|-|-|-|
|"dataFile"|string|Yes|-|Input data file. Must be either a tsv/csv file or it binarized version generated with the data2bin utility|
|"foldFile"|string|No|-|Optional input file containing the fold division of the dataset|
|"labelFile"|string|Yes, unless in simulation mode|-|Input file containing the labels of the examples of the dataset|
|"outFile"|string|Only in cv and test modes|-|Output file containing the output predictions|
|"forestDir"|string|Only in train and test modes|-|Output directory for saving / loading the trained models. Must be a valid directory on the filesystem|


##### "simulate"
```
"simulate": {
	"simulation": bool,
	"prob": float,
	"n": int,
	"m": int
},
```
Mandatory: no\
This field contains all the required information for enabling the internal dataset generator

| Name | Type | Mandatory | Default | Description |
|-|-|-|-|-|
|"simulation"|bool|No|false|If true, it enables the internal dataset generator. The fields "dataFile", "foldFile" and "labelFile" are ignored and a random dataset is generated|
|"prob"|float|only in simulation mode|-|This field represents the probability of generating a positive example. Must be a float in the [0,1] range, possibly very small for simulating highly unbalanced datasets|
|"n"|int|only in simulation mode|-|Number of examples to be generated|
|"m"|in|only in simulation mode|-|Number of features per sample to be generated|


##### "folds"
```
"folds": {
	"nFolds": int,
	"startingFold": int,
	"endingFold": int
}
```
Mandatory: Yes (No, if "foldFile" specified)\
This section specified the fold subdivision and on which fold execute the run.

| Name | Type | Mandatory | Default | Description |
|-|-|-|-|-|
|"nFolds"|int|Yes, unless a "foldFile" is specified|-|Specifies how many folds the dataset should be subdivided into. Ignored if "foldFile" has been declared|
|"startingFold"|int|No|-|These fields specify the starting and ending fold that parSMURF-NG have to evaluate. This is useful for parallelizing runs across different folds. If unspecified, parSMURF-NG performs the evaluation of the predictions on all folds|
|"startingFold"| | | | |

##### "params"
```
"params": {
	"nParts": array of int,
	"fp": array of int,
	"ratio": array of int,
	"k": array of int,
	"nTrees": array of int,
	"mtry": array of int
},
```
Mandatory: Yes\
This field contains the learning parameters for the run. All values must be passed as arrays.\
When "optimizer" is set to "no", only one combination is used for the run.\
When "optimizer" is set to "grid", parSMURF-NG generates all the possible hyper-parameter combinations and evaluate them in the internla CV / HO loop.\
For a deeper explanation of each parameter, please refer to the article

| Name | Type | Mandatory | Default | Description |
|-|-|-|-|-|
|"nParts"|array of int|Yes|10|Number of partitions (ensembles)|
|"fp"|array of int|Yes|1|Oversampling factor (0 disables over-sampling)|
|"ratio"|array of int|Yes|1|Undersampling factor (0 disables under-sampling)|
|"k"|array of int|Yes|5|Number of the nearest neighbors for SMOTE oversampling of the minority class|
|"nTrees"|array of int|Yes|10|Number of trees in each ensemble|
|"mtry"|array of int|Yes|sqrt(m)|mtry random forest parameter|


---

### Data format

As previously stated, data is provided to the application in two or three files.

##### Data file
This file should contain the set of samples and features needed for computing the predictions. It consists in an n x m matrix of double, where n is the number of examples and m the features. The matrix is read row-wise, i.e. :
```
   | m1   m2   m3   m4 ...
---------------------------
n1 | -------->
n2 |
n3 |
n4 |                          ( 1 )
.  |
.  |
.  |
```
As now, the file format is very strict, and data passed to the application should obey to these rules:\
- Each value must be separated by a single space. Each sample n should be entirely contained in a single line.
- Lines are terminated by a single new-line character ('\n').
- Empty fields are not allowed.
- The file must not contain an header.
The number of features is automatically detected.

We can consider to relax some of these restrictions in future releases. As now, the file format is almost the same as csv, but with no header and values are separated by spaces instead of commas.

For faster load time, it is possible to convert the input file to a binarized format with the data2bin utility. This utility accepts data files in the same format as the main parSMURF-NG application and converts it into a binary format which is more friendly to parSMURF-NG, as it represents the 1:1 memory dump of the dataset (no conversion must be done when importing from .bin files, hence loading time is massively reduced).


##### Label file
This file should contain the labelling of the examples. It consists in n space or new-line separated values, where n is the number of examples.\
It is a plain text file where each positive example is marked with "1" and negative examples with "0".

##### Fold file
This optional file should contain the fold subdivision. If specified, examples will be divided in folds according to this file. If not, a random stratified division will be performed. This file consists in n space or new-line separated integer values, where n is the number of examples.\
It is a plain text file where each number represents the fold to which each example is assigned. Fold numbering starts from "0" (zero).
Note that specifying the fold file name overrides the "nFolds" option in the configuration file.

##### Output file
Predictions will be saved as plain text file.\
Each line of this file contains the predicted value as probability; optionally it can contain the generated label (if in "simulation" mode) and the corresponding generated fold (if no fold file has been specified) as second and third value, respectively.\



---

### Random dataset generation

parSMURF-NG is provided with a random dataset generator for testing purposes.\
When enabled, a random dataset will be created. Sample features follow two normal distribution having the same variance but different average value, depending if an example falls in the positive or negative class.\
The user enables this mode by using the `"simulate: true"` option in the configuration file.\
The user is also forced to specify the probability that an example belongs to the minority class (`"prob": float`) and dimensionality of the dataset with the `"n": int` and `"m": int` options.\
An additional column will be added to the output file, containing the labelling that has been randomly generated according to the `"prob"` value.\
We also include the utility datasetGet to create the random dataset following the same rules (run datasetGen --help for additional information).

---

### Examples

A sample dataset and some configuration files are provided in the "samples" directory. The sample dataset has been created with the following command:
```
./datasetGen 20 980 15 5 2.5 data.txt label.txt 12
```
It consists of a 1000 samples, out of which 20 are marked as positive. Each sample is annotated with 15 features, out of which only 5 are informative (the rest are just random numbers).
We also provide a test set created with the following command:
```
./datasetGen 15 85 15 5 2.5 data_test.txt label_test.txt 32
```

**cfg_cv.json** performs a 5-fold cross validation on data.txt using the following parameters:/
```
"nParts": [4], "fp": [3], "ratio": [3], "k": [3], "nTrees": [25], "mtry": [0]
```
Execution is single-threaded. Results are saved on predictions_01.txt (3 columns, as folds are randomly generated).

**cfg_cv_mt.json** is the same task as cfg_cv.json, but partition computation is distributed over 4 threads

**cfg_train.json** treats data.txt entirely as training set. It trains a model and saves it on "savedModels" directory. The following parameters are used:
```
"nParts": [4], "fp": [3], "ratio": [3], "k": [3], "nTrees": [15], "mtry": [0]
```
No prediction is performed.

**cfg_predict.json** uses the saved model from cfg_train.json for predicting the values of the data_test.txt and label_test.txt.
It loads the model from the "savedModels" directory. Predictions are saved on "predictions_test.txt" file.

---



### License

This package is distributed under the GNU GPLv3 license. Please see the http://github.com/topopiccione/parSMURF-NG/COPYING file for the complete version of the license.

parSMURF-NG includes several third-party libraries which are distributed with their own license. In particular, source code of the following libraries is included in this package:

**ANN: Approximate Nearest Neighbor Searching**\
David M. Mount and Sunil Arya\
Version 1.1.2\
(https://www.cs.umd.edu/~mount/ANN/)\
Modified and redistributed under the GNU Lesser Public License v2.1\
Copy of the license is available in the src/ann_1.1.2 directory

**Ranger: A Fast Implementation of Random Forests**\
Marvin N. Wright\
Version 0.11.1\
(https://github.com/imbs-hl/ranger)\
Modified and redistributed under the MIT license\
Copy of the license is available in the src/ranger folder


Also, this software relies on several libraries whose source code is not included in the package, but it is automatically downloaded at compile time. These libraries are:

**Easylogging++**\
Zuhd Web Services\
(https://github.com/zuhd-org/easyloggingpp)\
Distributed under the MIT license\
Copy of the license is available at the project homepage

**Jsoncons**\
Daniel Parker\
(https://github.com/danielaparker/jsoncons)\
Distributed under the Boost license\
Copy of the license is available at the project homepage

**zlib**\
Jean-loup Gailly and Mark Adler\
(https://github.com/madler/zlib)\
Distributed under the zlib license\
Copy of the license is available at the project homepage
