# parSMURF-NG

This package contains parSMURF-NG, a High Performance Computing imbalance-aware machine learning tool for the genome-wide detection of pathogenic variants.

---

### Table of Contents
<pre>
<a href="#Overview">Overview</a>
<a href="#Licensa">License</a>
</pre>

---

### Overview
parSMURF-NG is a fast and scalable C++ implementation of the HyperSMURF algorithm - hyper-ensemble of SMOTE Undersampled Random Forests - an ensemble approach explicitly designed to deal with the huge imbalance between deleterious and neutral variants.

The algorithm is outlined in the following papers:\
A. Petrini, M. Mesiti, M. Schubach, M. Frasca, D. Danis, M. Re, G. Grossi, T. Castrignanò, P. N. Robinson and G. Valentini, "parSMURF, a High Performance Computing tool for the genome-wide detection of pathogenic variants OR parSMURF, a High Performance Computing tool for supervised big data analysis in Genomic Medicine"\
<TBD>

Schubach, Matteo Re, Peter N. Robinson & Giorgio Valentini, "Imbalance-Aware Machine Learning for Predicting Rare and Common Disease-Associated Non-Coding Variants", Scientific Reports, 2017/06/07\
https://www.nature.com/articles/s41598-017-03011-5

parSMURF-NG is a more stable, faster, more scalable and up-to-date version of parSMURF. Also, usability has been improved.

Proper documentation will be published as soon as possible.

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
