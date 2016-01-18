#ifndef ATSPECTRUM_HPP
#define ATSPECTRUM_HPP

#include <cstdlib>
#include <cstdio>
#include <vector>
#include <arpack++/arlnsmat.h>
#include <arpack++/arlssym.h>
#include <ATSuite/transferOperator.hpp>

/** \file transferSpectrum.hpp
 *  \brief Get spectrum of transferOperator using ARPACK++.
 *   
 *  Analyse the spectrum of the forward and backward transition matrices
 *  of a transferOperator object using ARPACK++.
 */


/*
 * Class declarations
 */

/**
 * \brief Utility class used to give configuration options to ARPACK++.
 * 
 * Utility class used to give configuration options to ARPACK++.
 */
class configAR {
public:
  std::string which_;       //!< Which eigenvalues to look for. 'LM' for Largest Magnitude
  int ncv_ = 0;             //!< The number of Arnoldi vectors generated at each iteration of ARPACK
  double tol_ = 0.;         //!< The relative accuracy to which eigenvalues are to be determined
  int maxit_ = 0;           //!< The maximum number of iterations allowed
  double *resid_ = NULL;    //!< A starting vector for the Arnoldi process
  bool AutoShift_ = true;   //!< Shifts for the implicit restarting of the Arnoldi method


  /** Constructor. */
  configAR(const std::string& which="LM", int ncv=0, double tol=0.,
	   int maxit=0, double *resid=NULL, bool AutoShift=true);
};
/** Declare default class looking for largest magnitude eigenvalues */
configAR defaultCfgAR ("LM");


/** \brief Transfer operator spectrum.
 *
 *  Class used to calculate the spectrum of forward and backward
 *  transition matrices of a transferOperator object.
 */
class transferSpectrum {
public:
  int nev = 0;                  //!< Number of eigenvalues and vectors to search;
  double *EigValForwardReal;    //!< Real part of eigenvalues of forward transition matrix
  double *EigValForwardImag;    //!< Imaginary part of eigenvalues of forward transition matrix
  double *EigValBackwardReal;   //!< Real part of eigenvalues of backward transition matrix
  double *EigValBackwardImag;   //!< Imaginary part of eigenvalues of backward transition matrix
  double *EigVecForward;        //!< Eigenvectors of forward transition matrix
  double *EigVecBackward;       //!< Eigenvectors of backward transition matrix
  transferOperator *transferOp; //!< Transfer operator of the eigen problem


  /** \brief Default constructor. */
  transferSpectrum(){}
  /** \brief Constructor allocating for nev_ eigenvalues and vectors. */
  transferSpectrum(const int nev_, transferOperator *transferOp_);
  /** \brief Destructor desallocating. */
  ~transferSpectrum();


  /** \brief Get transfer operator spectrum. */
  int getSpectrum(const configAR cfgAR=defaultCfgAR);
  

  /** \brief Write complex eigenvalues and eigenvectors from ARPACK++. */
  int writeSpectrum(const char *EigValForwardFile, const char *EigVecForwardFile,
		    const char *EigValBackwardFile, const char *EigVecBackwardFile);
};


/*
 *  Functions declarations
 */
/** \brief Get spectrum of a nonsymmetric matrix using ARPACK++. */
int getSpectrumAR(ARluNonSymMatrix<double, double> *M, int nev, configAR cfgAR,
		  double *EigValReal, double *EigValImag, double *EigVec);
int writeSpectrumAR(FILE *fEigVal, FILE *fEigVec,
		    const double *EigValReal, const double *EigValImag,
		    const double *EigVec, const int nev, const size_t N);


/*
 * Constructors and destructors definitions
 */

/**
 *  Constructor of configAR with default parameters.
 */
configAR::configAR(const std::string& which, int ncv, double tol,
		   int maxit, double *resid, bool AutoShift)
{
  which_ = which;
  ncv_ = ncv;
  tol_ = tol;
  maxit_ = maxit;
  resid_ = resid;
  AutoShift_ = AutoShift;
}


/**
 * Constructor allocating space for a given number of eigenvalues and vectors
 * for a given transferOperator.
 * \param[in] nev_        Number of eigenvalues and eigenvectors for which to allocate.
 * \param[in] transferOp_ Pointer to the transferOperator on which to solve the eigen problem.
 */
transferSpectrum::transferSpectrum(const int nev_, transferOperator *transferOp_)
{
  /** Number of eigenvalues to search and transferOperator */
  nev = nev_;
  transferOp = transferOp_;
  
  /** Allocate for possible additional eigenvalue of complex pair */
  EigValForwardReal = new double [nev+1];
  EigValForwardImag = new double [nev+1];
  EigValBackwardReal = new double [nev+1];
  EigValBackwardImag = new double [nev+1];
  
  /** Allocate for both real and imaginary part
   *  but only for one member of the pair */
  EigVecForward = new double [(nev+2) * transferOp->N];
  EigVecBackward = new double [(nev+2) * transferOp->N];
}


/**
 * Destructor desallocates. 
 */
transferSpectrum::~transferSpectrum()
{
  delete[] EigValForwardReal;
  delete[] EigValForwardImag;
  delete[] EigVecForward;
  delete[] EigValBackwardReal;
  delete[] EigValBackwardImag;
  delete[] EigVecBackward;
}


/*
 * Methods definitions
 */

/**
 * Get spectrum of transfer operator,
 * including the complex eigenvalues,
 * the left eigenvectors of the forward transition matrix
 * and the right eigenvectors of the backward transitioin matrix.
 * The vectors of eigenvalues and eigenvectors should not be preallocated.
 */
int
transferSpectrum::getSpectrum(configAR cfgAR)
{
  int nconv, idx;
  ARluNonSymMatrix<double, double> *mAR;
  int *irow, *pcol;
  
  /** Check if constructor has been called */
  if (!nev) {
    fprintf(stderr, "Constructor with argument a positive number of eigenvalues \
to search and a transferOperator should be called before to get spectrum.\n");
    return EXIT_FAILURE;
  }

  /** Get transpose of forward transition matrix in ARPACK CCS format.
   *  Transposing is trivial since the transition matrix is in CRS format.
   *  However, indices should be converted from size_t to int.
   *  Use order = 2 for degree ordering of A.T + A */
  mAR = new ARluNonSymMatrix<double, double>;
  irow = (int *) malloc(transferOp->P->nz * sizeof(int));
  pcol = (int *) malloc((transferOp->N + 1) * sizeof(int));
  for (idx = 0; idx < transferOp->P->nz; idx++)
    irow[idx] = (int) transferOp->P->innerIdx[idx];
  for (idx = 0; idx < transferOp->N + 1; idx++)
    pcol[idx] = (int) transferOp->P->p[idx];
  mAR->DefineMatrix(transferOp->N, transferOp->P->nz, transferOp->P->data,
		    irow, pcol, 0.1, 2, true);

  /** Get eigenvalues and vectors of forward transition matrix */
  nconv = getSpectrumAR(mAR, nev, cfgAR, EigValForwardReal, EigValForwardImag, EigVecForward);
  free(irow);
  free(pcol);
  delete mAR;

  /** Get transpose of backward transition matrix in ARPACK CCS format */
  mAR = new ARluNonSymMatrix<double, double>;
  irow = (int *) malloc(transferOp->Q->nz * sizeof(int));
  pcol = (int *) malloc((transferOp->N + 1) * sizeof(int));
  for (idx = 0; idx < transferOp->Q->nz; idx++)
    irow[idx] = (int) transferOp->Q->innerIdx[idx];
  for (idx = 0; idx < transferOp->N + 1; idx++)
    pcol[idx] = (int) transferOp->Q->p[idx];
  mAR->DefineMatrix(transferOp->N, transferOp->Q->nz, transferOp->Q->data,
		    irow, pcol, 0.1, 2, true);
  
  /** Get eigenvalues and vectors of backward transition matrix */
  nconv += getSpectrumAR(mAR, nev, cfgAR, EigValBackwardReal, EigValBackwardImag, EigVecBackward);
  free(irow);
  free(pcol);
  delete mAR;
  
  return nconv;
}
    

/**
 * Write complex eigenvalues and eigenvectors
 * of forward and backward transition matrices of a transfer operator to file.
 * \param[in] EigValForwardFile  File name of the file to print forward eigenvalues.
 * \param[in] EigVecForwardFile  File name of the file to print forward eigenvectors.
 * \param[in] EigValBackwardFile File name of the file to print backward eigenvalues.
 * \param[in] EigVecBackwardFile File name of the file to print backward eigenvectors.
 * \return                       Exit status.
 */
int
transferSpectrum::writeSpectrum(const char *EigValForwardFile, const char *EigVecForwardFile,
				const char *EigValBackwardFile, const char *EigVecBackwardFile)
{
  FILE *streamEigVal, *streamEigVec;
  
  /** Open files for forward */
  if (!(streamEigVal = fopen(EigValForwardFile, "w"))){
    fprintf(stderr, "Can't open %s for writing forward eigenvalues",
	    EigValForwardFile);
    perror("");
    return EXIT_FAILURE;
  }
  if (!(streamEigVec = fopen(EigVecForwardFile, "w"))){
    fprintf(stderr, "Can't open %s for writing forward eigenvectors",
	    EigVecForwardFile);
    perror("");
    return EXIT_FAILURE;
  }

  /** Write forward */
  if (writeSpectrumAR(streamEigVal, streamEigVec,
		      EigValForwardReal, EigValForwardImag, EigVecForward,
		      nev, transferOp->N)) {
    fprintf(stderr, "Error writing spectrum.\n");
    return EXIT_FAILURE;
  }

  /** Close */
  fclose(streamEigVal);
  fclose(streamEigVec);

  /** Open files for backward */
  if (!(streamEigVal = fopen(EigValBackwardFile, "w"))){
    fprintf(stderr, "Can't open %s for writing backward eigenvalues",
	    EigValBackwardFile);
    perror("");
    return EXIT_FAILURE;
  }
  if (!(streamEigVec = fopen(EigVecBackwardFile, "w"))){
    fprintf(stderr, "Can't open %s for writing backward eigenvectors",
	    EigVecBackwardFile);
    perror("");
    return EXIT_FAILURE;
  }

  /** Write backward */
  if (writeSpectrumAR(streamEigVal, streamEigVec,
		      EigValBackwardReal, EigValBackwardImag, EigVecBackward,
		      nev, transferOp->N)) {
    fprintf(stderr, "Error writing spectrum.\n");
    return EXIT_FAILURE;
  }

  /** Close */
  fclose(streamEigVal);
  fclose(streamEigVec);
  
  return 0;
}
/*
 * Functions definitions
 */

/**
 *
 * Get spectrum of a nonsymmetric matrix using ARPACK++.
 * \param[in] P ARPACK++ CSC sparse matrix from which to calculate the spectrum.
 * \param[in] nev Number of eigenvalues and eigenvectors to find.
 * \param[in] cfgAR Configuration options passed as a configAR object.
 * \param[out] EigValReal Real part of found eigenvalues.
 * \param[out] EigValImag Imaginary  part of found eigenvalues.
 * \param[out] EigVec Found eigenvectors.
 * \return Number of eigenvalues and eigenvectors found.
 */
int
getSpectrumAR(ARluNonSymMatrix<double, double> *M, int nev, configAR cfgAR,
	      double *EigValReal, double *EigValImag, double *EigVec)
{
  ARluNonSymStdEig<double> EigProb;
  int nconv;

  // Define eigen problem
  EigProb = ARluNonSymStdEig<double>(nev, *M, cfgAR.which_, cfgAR.ncv_, cfgAR.tol_,
				     cfgAR.maxit_, cfgAR.resid_, cfgAR.AutoShift_);
  
  // Find eigenvalues and left eigenvectors
  EigProb.EigenValVectors(EigVec, EigValReal, EigValImag);
  nconv = EigProb.ConvergedEigenvalues();


  return nconv;
}


/**
 * Write complex eigenvalues and eigenvectors obtained as arrays from ARPACK++.
 * \param[in] fEigVal    File descriptor for eigenvalues.
 * \param[in] fEigVec    File descriptor for eigenvectors.
 * \param[in] EigValReal Array of eigenvalues real parts.
 * \param[in] EigValImag Array of eigenvalues imaginary parts.
 * \param[in] EigVec     Array of eigenvectors.
 * \param[in] nev        Number of eigenvalues and eigenvectors.
 * \param[in] N          Length of the eigenvectors.
 * \return               Exit status.
 */
int
writeSpectrumAR(FILE *fEigVal, FILE *fEigVec,
		const double *EigValReal, const double *EigValImag,
		const double *EigVec, const int nev, const size_t N)
{
  size_t vecCount = 0;
  int ev =0;
  // Write real and imaginary parts of each eigenvalue on each line
  // Write on each pair of line the real part of an eigenvector then its imaginary part
  while (ev < nev) {
    // Always write the eigenvalue
    fprintf(fEigVal, "%lf %lf\n", EigValReal[ev], EigValImag[ev]);
    // Always write the real part of the eigenvector ev
    for (size_t i = 0; i < N; i++){
      fprintf(fEigVec, "%lf ", EigVec[vecCount*N+i]);
    }
    fprintf(fEigVec, "\n");
    vecCount++;
    
    // Write its imaginary part or the zero vector
    if (EigValImag[ev] != 0.){
      for (size_t i = 0; i < N; i++)
	fprintf(fEigVec, "%lf ", EigVec[vecCount*N+i]);
      vecCount++;
      // Skip the conjugate
      ev += 2;
    }
    else{
      for (size_t i = 0; i < N; i++)
	fprintf(fEigVec, "%lf ", 0.);
      ev += 1;
    }
    fprintf(fEigVec, "\n");
  }

  /** Check for printing errors */
  if (ferror(fEigVal)) {
    fprintf(stderr, "Error printing eigenvalues.\n");
    return EXIT_FAILURE;
  }
  if (ferror(fEigVec)) {
    fprintf(stderr, "Error printing eigenvectors.\n");
    return EXIT_FAILURE;
  }

  return 0;
}
	
#endif
