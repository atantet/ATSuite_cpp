#ifndef ATSPECTRUM_HPP
#define ATSPECTRUM_HPP

#include <cstdlib>
#include <cstdio>
#include <vector>
#include <Eigen/Sparse>
#include <arpack++/arlnsmat.h>
#include <arpack++/arlssym.h>
#include <ATSuite/atio.hpp>

/** \file atspectrum.hpp
 *  \brief Get spectrum of sparse matrices using ARPACK++.
 *   
 *  ATSuite functions to get spectrum of sparse matrices using ARPACK++.
 *  Also includes reading and conversion routines to ARPACK++ CSC matrices.
 */

// Typedef declaration
/** \brief Eigen sparse CSR matrix of double type. */
typedef SparseMatrix<double, RowMajor> SpMatCSR;
/** \brief Eigen sparse CSC matrix of double type. */
typedef SparseMatrix<double, ColMajor> SpMatCSC;

// Class declarations
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
  configAR(const std::string&, int, double, int, double *, bool);
};

// Function declarations 
ARluNonSymMatrix<double, double> * pajek2AR(FILE *);
ARluNonSymMatrix<double, double> * Eigen2AR(SpMatCSC *);
ARluNonSymMatrix<double, double> * Eigen2AR(SpMatCSR *);
ARluSymMatrix<double> * Eigen2ARSym(SpMatCSC *);
ARluSymMatrix<double> * Eigen2ARSym(SpMatCSR *);
ARluNonSymMatrix<double, double> * Compressed2AR(FILE *);
int getSpectrum(ARluNonSymMatrix<double, double> *, int, configAR, double *, double *, double *);
void writeSpectrum(FILE *, FILE *, double *, double *, double *, int, size_t);

// Definitions
/**
 * \brief Main constructor.
 * 
 * Main constructor with default parameters.
 */
configAR::configAR(const std::string& which="LM", int ncv=0, double tol=0.,
		   int maxit=0, double *resid=NULL, bool AutoShift=true)
{
  which_ = which;
  ncv_ = ncv;
  tol_ = tol;
  maxit_ = maxit;
  resid_ = resid;
  AutoShift_ = AutoShift;
}

/**
 * \brief Scans an ARPACK++ nonsymmetric matrix from a Pajek file.
 *
 * Scans an ARPACK++ LU nonsymmetric CSC matrix
 * (see <a href="http://www.caam.rice.edu/software/ARPACK/arpack++.html">ARPACK++ documentation</a>)
 * from a Pajek file
 * (see <a href="http://mrvar.fdv.uni-lj.si/pajek/">Pajek documentation</a>).
 * \param[in] fp    Descriptor of the file to which to scan.
 * \return    ARPACK++ LU nonsymmetric matrix scanned.
 */
ARluNonSymMatrix<double, double> *pajek2AR(FILE *fp)
{
  // The edge entries must be ordered by row and then by col.
  char label[20];
  int N, E, row, col;
  double val;
  int *irow, *pcol;
  double *nzval;
  ARluNonSymMatrix<double, double> *T = new ARluNonSymMatrix<double, double>;

  // Read vertices
  fscanf(fp, "%s %d", label, &N);
  pcol = new int [N+1];

  // Read first (assume monotonous)
  for (int k = 0; k < N; k++)
    fscanf(fp, "%d %s", &row, label);

  // Read number of edges
  fscanf(fp, "%s %d", label, &E);
  irow = new int [E];
  nzval = new double [E];
  for (int k = 0; k < N+1; k++)
    pcol[k] = E;

  // Read edges
  for (int k = 0; k < E; k++){
    fscanf(fp, "%d %d %lf", &row, &col, &val);
    irow[k] = row;
    printf("irow[%d] = %d\n", k, row);
    nzval[k] = val;
    if (k < pcol[col])
      pcol[col] = k;
  }

  // Define matrix, order=2 for degree ordering of A.T + A (good for transition mat)
  T->DefineMatrix(N, E, nzval, irow, pcol, 0.1, 2, true);

  return T;
}

/**
 * \brief Scans an ARPACK++ nonsymmetric matrix from a file in compressed format.
 *
 * Scans an ARPACK++ LU nonsymmetric CSC matrix
 * (see <a href="http://www.caam.rice.edu/software/ARPACK/arpack++.html">ARPACK++ documentation</a>)
 * from a matrix file in compressed format (see atio.hpp documentation).
 * \param[in] fp    Descriptor of the file to which to scan.
 * \return    ARPACK++ LU nonsymmetric matrix scanned.
 */
ARluNonSymMatrix<double, double> * Compressed2AR(FILE *fp)
{
  int innerSize, outerSize, nnz;
  char sparseType[4];
  double *nzval;
  int *irow, *pcol;
  ARluNonSymMatrix<double, double> *T = new ARluNonSymMatrix<double, double>;

  // Read type, inner size, outer size and number of non-zeros and allocate
  fscanf(fp, "%s %d %d %d", sparseType, &innerSize, &outerSize, &nnz);
  nzval = new double [nnz];
  irow = new int [nnz];
  pcol = new int[outerSize+1];

  // Read values
  for (int nz = 0; nz < nnz; nz++)
    fscanf(fp, "%lf ", &nzval[nz]);

  // Read row indices
  for (int nz = 0; nz < nnz; nz++)
    fscanf(fp, "%d ", &irow[nz]);

  // Read first element of column pointer
  for (int outer = 0; outer < outerSize+1; outer++)
    fscanf(fp, "%d ", &pcol[outer]);
  
  // Define matrix, order=2 for degree ordering of A.T + A (good for transition mat)
  T->DefineMatrix(outerSize, nnz, nzval, irow, pcol, 0.1, 2, true);
  
  return T;
}

/**
 * \brief Converts an Eigen CSC matrix to an ARPACK++ nonsymmetric CSC matrix.
 *
 * Converts an Eigen CSC matrix to ARPACK++ LU nonsymmetric CSC matrix.
 * \param[in] TEigen    Eigen matrix from which to convert.
 * \return ARPACK++ LU nonsymmetrix CSC matrix converted.
 */
ARluNonSymMatrix<double, double>* Eigen2AR(SpMatCSC *TEigen)
{
  int outerSize, nnz;
  double *nzval;
  int *irow, *pcol;
  ARluNonSymMatrix<double, double> *T = new ARluNonSymMatrix<double, double>;

  outerSize = TEigen->outerSize();
  nnz = TEigen->nonZeros();
  nzval = new double [nnz];
  irow = new int [nnz];
  pcol = new int [outerSize+1];

  // Set values
  nzval = TEigen->valuePtr();

  // Set inner indices
  irow = TEigen->innerIndexPtr();

  // Set first element of column pointer
  pcol = TEigen->outerIndexPtr();

  // Define matrix, order=2 for degree ordering of A.T + A (good for transition mat)
  T->DefineMatrix(outerSize, nnz, nzval, irow, pcol, 0.1, 2, true);
  
  return T;
}

/**
 * \brief Converts an Eigen CSR matrix to an ARPACK++ nonsymmetric CSC matrix.
 *
 * Converts an Eigen CSR matrix to an ARPACK++ LU nonsymmetric CSC matrix.
 * \param[in] TEigenCSR    Eigen matrix from which to convert.
 * \return ARPACK++ LU nonsymmetrix CSC matrix converted.
 */
ARluNonSymMatrix<double, double>* Eigen2AR(SpMatCSR *TEigenCSR)
{
  SpMatCSC *TEigen;
  ARluNonSymMatrix<double, double> *T;

  // Convert from Eigen CSR to CSC
  TEigen = CSR2CSC(TEigenCSR);

  // Convert from Eigen CSC to AR
  T = Eigen2AR(TEigen);

  return T;
}

/**
 * \brief Converts an Eigen CSC matrix to an ARPACK++ symmetric CSC matrix.
 *
 * Converts an Eigen CSC matrix to ARPACK++ LU symmetric matrix.
 * \param[in] TEigen    Eigen matrix from which to convert.
 * \return ARPACK++ LU symmetrix CSC matrix converted.
 */
ARluSymMatrix<double>* Eigen2ARSym(SpMatCSC *TEigen)
{
  int outerSize, nnz;
  double *nzval, *nzvalSym;
  int *irow, *pcol, *irowSym, *pcolSym;
  ARluSymMatrix<double> *T = new ARluSymMatrix<double>;
  int nzIdx, nzSymIdx, innerIdx;
  bool isNewCol;

  outerSize = TEigen->outerSize();
  nnz = TEigen->nonZeros();
  nzval = new double [nnz];
  irow = new int [nnz];
  pcol = new int [outerSize+1];
  nzvalSym = new double [(int) ((nnz-outerSize)/2)+outerSize+1];
  irowSym = new int [(int) ((nnz-outerSize)/2)+outerSize+1];
  pcolSym = new int [outerSize+1];

  // Set values
  nzval = TEigen->valuePtr();
  // Set inner indices
  irow = TEigen->innerIndexPtr();
  // Set first element of column pointer
  pcol = TEigen->outerIndexPtr();

  // Discard lower triangle
  nzIdx = 0;
  nzSymIdx = 0;
  for (int outerIdx = 0; outerIdx < outerSize; outerIdx++){
    isNewCol = true;
    while (nzIdx < pcol[outerIdx+1]){
      innerIdx = irow[nzIdx];
      if (outerIdx >= innerIdx){
	nzvalSym[nzSymIdx] = nzval[nzIdx];
	irowSym[nzSymIdx] = irow[nzIdx];
	if (isNewCol){
	  pcolSym[outerIdx] = nzSymIdx;
	  isNewCol = false;
	}
	nzSymIdx++;
      }
      nzIdx++;
    }
    // Check for empty column
    if (isNewCol) 
      pcolSym[outerIdx] = nzSymIdx;
  }
  pcolSym[outerSize] = nzSymIdx;

  // Define matrix, order=2 for degree ordering of A.T + A (good for transition mat)
  T->DefineMatrix(outerSize, nzSymIdx, nzvalSym, irowSym, pcolSym,
		  'U', 0.1, 2, true);
  
  return T;
}

/**
 * \brief Converts an Eigen CSR matrix to an ARPACK++ symmetric CSC matrix.
 *
 * Converts an Eigen CSR matrix to ARPACK++ LU symmetric matrix.
 * \param[in] TEigenCSR    Eigen matrix from which to convert.
 * \return ARPACK++ LU symmetrix CSC matrix converted.
 */
ARluSymMatrix<double>* Eigen2ARSym(SpMatCSR *TEigenCSR)
{
  SpMatCSC *TEigen;
  ARluSymMatrix<double> *T;

  // Convert from Eigen CSR to CSC
  TEigen = CSR2CSC(TEigenCSR);

  // Convert from Eigen CSC to AR
  T = Eigen2ARSym(TEigen);

  return T;
}

/**
 * \brief Get spectrum of a nonsymmetric matrix using ARPACK++.
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
getSpectrum(ARluNonSymMatrix<double, double> *P, int nev, configAR cfgAR,
	    double *EigValReal, double *EigValImag, double *EigVec)
{
  ARluNonSymStdEig<double> EigProb;
  int nconv;

  // Define eigen problem
  EigProb = ARluNonSymStdEig<double>(nev, *P, cfgAR.which_, cfgAR.ncv_, cfgAR.tol_,
				     cfgAR.maxit_, cfgAR.resid_, cfgAR.AutoShift_);
  
  // Find eigenvalues and left eigenvectors
  EigProb.EigenValVectors(EigVec, EigValReal, EigValImag);
  nconv = EigProb.ConvergedEigenvalues();


  return nconv;
}

/**
 * \brief Write complex eigenvalues and eigenvectors from ARPACK++.
 * 
 * Write complex eigenvalues and eigenvectors obtained as arrays from ARPACK++.
 * \param[in] fEigVal File descriptor for eigenvalues.
 * \param[in] fEigVec File descriptor for eigenvectors.
 * \param[in] EigValReal Array of eigenvalues real parts.
 * \param[in] EigValImag Array of eigenvalues imaginary parts.
 * \param[in] EigVec Array of eigenvectors.
 * \param[in] nev Number of eigenvalues and eigenvectors.
 * \param[in] N Length of the eigenvectors.
 */
void writeSpectrum(FILE *fEigVal, FILE *fEigVec,
		   double *EigValReal, double *EigValImag,
		   double *EigVec, int nev, size_t N)
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

  return;
}
	
#endif
