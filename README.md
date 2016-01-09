Introduction               {#introduction}
============

ATSuite C++ is a collection of scientific routines in C++
originally developed by Alexis Tantet for _research purpose_.
These codes are open source in order to promote reproducibility.
Visit Alexis' home page [UU] for contact.
The full documentation can be found at [ATSuite_cpp_doc].


Installation               {#installation}
============

Getting the code
----------------

First the ATSuite_cpp repository should be cloned using [git].
To do so:
1. Change directory to where you want to clone the repository $GITDIR:

        cd $GITDIR
     
2. Clone the ATSuite_cpp repository (git should be installed):

        git clone https://github.com/atantet/ATSuite_cpp
     
Dependencies                    {#dependencies}
------------

- [GSL] is used as the main C scientific library.
- [Eigen] is a C++ template library for linear algebra, mainly used for sparse matrices manipulation.
- [ARPACK++] is an object-oriented version of the ARPACK package, mainly used to find the leading part of the spectrum of sparse matrices.

Installing the code
-------------------

1. Create a directory ATSuite/ in your favorite include directory $INCLUDE:

        mkdir $INCLUDE/ATSuite
     
2. Copy the ATSuite_cpp/*.hpp source files to $INCLUDE/ATSuite/:

        cd $GITDIR/ATSuite_cpp
        cp *.hpp $INCLUDE/ATSuite
     
3. Include these files in your C++ codes. For example, in order to include the matrix manipulation functions in atmatrix.hpp,
add in your C++ file:

        #include <ATSuite/atmatrix.hpp>
    

Updating the code
-----------------

1. Pull the ATSuite_cpp repository:

        cd $GITDIR/ATSuite_cpp     
        git pull
     
2. Copy the source files to your favorite include directory $INCLUDE:

        cp *.hpp $INCLUDE/ATSuite


Disclaimer            {#disclaimer}
==========

These codes are developed for _research purpose_.
_No warranty_ is given regarding their robustess.

[UU]: http://www.uu.nl/staff/AJJTantet/ "Alexis' personal page"
[git]: https://git-scm.com/ "git"
[ATSuite_cpp_doc]: http://atantet.github.io/ATSuite_cpp/ "ATSuite C++ documentation"
[GSL]: http://www.gnu.org/software/gsl/ "GSL - GNU Scientific Library"
[Eigen]: http://eigen.tuxfamily.org/ "Eigen"
[ARPACK++]: http://www.caam.rice.edu/software/ARPACK/arpack++.html "ARPACK++"