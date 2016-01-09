#ifndef ATSPECTRUM_HPP
#define ATSPECTRUM_HPP

#include <vector>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include "arlnsmat.h"
#include "arlssym.h"
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <ATSuite/atio.hpp>

/** \file atspectrum.hpp
 *  \brief Get spectrum of sparse matrices using Arpack++.
 *   
 *  ATSuite functions to get spectrum of sparse matrices using Arpack++.
 *  Also includes reading and conversion routines to Arpack++ CSC matrices.
 */

// Typedef declaration
/** \brief Eigen sparse CSR matrix of double type. */
typedef SparseMatrix<double, RowMajor> SpMatCSR;
/** \brief Eigen sparse CSC matrix of double type. */
typedef SparseMatrix<double, ColMajor> SpMatCSC;

// Declarations
ARluNonSymMatrix<double, double> * pajek2AR(FILE *);
ARluNonSymMatrix<double, double> * Eigen2AR(SpMatCSC *);
ARluNonSymMatrix<double, double> * Eigen2AR(SpMatCSR *);
ARluSymMatrix<double> * Eigen2ARSym(SpMatCSC *);
ARluSymMatrix<double> * Eigen2ARSym(SpMatCSR *);
ARluNonSymMatrix<double, double> * Compressed2AR(FILE *);

// Definitions
/**
 * Scans an Arpack++ LU nonsymmetric CSC matrix
 * (see <a href="http://www.caam.rice.edu/software/ARPACK/arpack++.html">ARPACK++ documentation</a>)
 * from a Pajek file
 * (see <a href="http://mrvar.fdv.uni-lj.si/pajek/">Pajek documentation</a>).
 * \param[in] fp    Descriptor of the file to which to scan.
 * \return    Arpack++ LU nonsymmetric matrix scanned.
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
 * Scans an Arpack++ LU nonsymmetric CSC matrix
 * (see <a href="http://www.caam.rice.edu/software/ARPACK/arpack++.html">ARPACK++ documentation</a>)
 * from a matrix file in compressed format (see atio.hpp documentation).
 * \param[in] fp    Descriptor of the file to which to scan.
 * \return    Arpack++ LU nonsymmetric matrix scanned.
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
 * Converts an Eigen CSC matrix to Arpack++ LU nonsymmetric matrix.
 * \param[in] TEigen    Eigen matrix from which to convert.
 * \return Arpack++ LU nonsymmetrix CSC matrix converted.
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
 * Converts an Eigen CSR matrix to Arpack++ LU nonsymmetric matrix.
 * \param[in] TEigen    Eigen matrix from which to convert.
 * \return Arpack++ LU nonsymmetrix CSC matrix converted.
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
 * Converts an Eigen CSC matrix to Arpack++ LU symmetric matrix.
 * \param[in] TEigen    Eigen matrix from which to convert.
 * \return Arpack++ LU symmetrix CSC matrix converted.
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
 * Converts an Eigen CSR matrix to Arpack++ LU symmetric matrix.
 * \param[in] TEigen    Eigen matrix from which to convert.
 * \return Arpack++ LU symmetrix CSC matrix converted.
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

#endif
