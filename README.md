Introduction    {#introduction}
============

ATSuite C++ is a collection of scientific routines in C++
originally developed by Alexis Tantet for _research purpose_.
These codes are open source in order to promote reproducibility.
Visit Alexis' home page [UU] for contact.


Installation    {#installation}
============

Getting the code
----------------

First the ATSuite_cpp repository should be cloned using [git].
To do so:
1. Change directory to where you want to clone the repository $GITDIR:
    cd $GITDIR
2. Clone the ATSuite_cpp repository (git should be installed):
    git clone https://github.com/atantet/ATSuite_cpp

Installing the code
-------------------

1. Create a directory ATSuite/ in your favorite include directory $INCLUDE:
    mkdir $INCLUDE/ATSuite
2. Copy the ATSuite_cpp/*.hpp source files to $INCLUDE/ATSuite/:
    cd $GITDIR/ATSuite_cpp
    cp *.hpp $INCLUDE/ATSuite
3. Include these files in your C++ codes.
For example, in order to include the matrix manipulation functions in atmatrix.hpp,
add:
    #include <ATSuite/atmatrix.hpp>
in your C++ file.

Updating the code
-----------------

1. Pull the ATSuite_cpp repository:
    cd $GITDIR/ATSuite_cpp
    git pull
2. Copy the source files to your favorite include directory $INCLUDE:
    cp *.hpp $INCLUDE/ATSuite


Disclaimer    {#disclaimer}
==========

These codes are developed for _research purpose_.
No warranty is given regarding their robustess.

[UU]: http://www.uu.nl/staff/AJJTantet/ "Alexis' personal page"
[git]: https://git-scm.com/ "git"