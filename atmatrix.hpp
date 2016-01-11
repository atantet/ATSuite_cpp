#ifndef ATMATRIX_HPP
#define ATMATRIX_HPP

#include <vector>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_vector_int.h>
#include <gsl/gsl_matrix.h>
#include <Eigen/Dense>
#include <Eigen/Sparse>

/** \file atmatrix.hpp
 *  \brief Matrix manipulation.
 *   
 *  ATSuite matrix manipulation routines.
 */

// Typedef definitions
/** \brief Eigen CSR matrix of double type. */
typedef Eigen::SparseMatrix<double, Eigen::RowMajor> SpMatCSR;
/** \brief Eigen CSR matrix of integer type. */
typedef Eigen::SparseMatrix<int, Eigen::RowMajor> SpMatIntCSR;
/** \brief Eigen CSC matrix of double type. */
typedef Eigen::SparseMatrix<double, Eigen::ColMajor> SpMatCSC;
/** \brief Eigen CSC matrix of boolean type. */
typedef Eigen::SparseMatrix<bool, Eigen::ColMajor> SpMatCSCBool;

// Declarations
gsl_vector* getRowSum(SpMatCSR *);
void getRowSum(SpMatCSR *, gsl_vector *);
gsl_vector_int* getRowSum(SpMatIntCSR *);
gsl_vector* getColSum(SpMatCSR *);
void getColSum(SpMatCSR *, gsl_vector *);
gsl_vector* getColSum(SpMatCSC *);
double getSum(SpMatCSR *);
double sumVectorElements(gsl_vector *);
void normalizeVector(gsl_vector *);
void normalizeRows(SpMatCSR *, gsl_vector *);
SpMatCSCBool * cwiseGT(SpMatCSC *, double);
SpMatCSCBool * cwiseLT(SpMatCSC *, double);
bool any(SpMatCSCBool *);
double max(SpMatCSC *);
double min(SpMatCSC *);
std::vector<int> * argmax(SpMatCSC *);
std::vector<int> * argmin(SpMatCSC *);


// Definitions
/**
 * \brief Get the sum of each row of an Eigen CSR matrix.
 *
 * Get the sum of each row of an Eigen CSR matrix as a GSL vector.
 * \param[in] T    Eigen CSR matrix on which to sum.
 * \return GSL vector of the sum of the rows of the sparse matrix.
 */
gsl_vector*
getRowSum(SpMatCSR *T)
{
  int N = T->rows();
  gsl_vector *rowSum = gsl_vector_calloc(N);

  // Calculate row sums
  for (int i = 0; i < N; i++)
    for (SpMatCSR::InnerIterator it(*T, i); it; ++it)
      gsl_vector_set(rowSum, i, gsl_vector_get(rowSum, i)+it.value()); 
  
  return rowSum;
}

/** 
 * \brief Get the sum of each row of an Eigen CSR matrix.
 *
 * Get the sum of each row of an Eigen CSR matrix as a GSL vector
 * (alternate version with output as argument).
 * \param[in] T    Eigen CSR matrix on which to sum.
 * \param[out] rowSum    GSL vector of the sum of rows of the sparse matrix.
 */
void
getRowSum(SpMatCSR *T, gsl_vector *rowSum)
{
  int N = T->rows();

  // Calculate row sums
  gsl_vector_set_all(rowSum, 0.);
  for (int i = 0; i < N; i++)
    for (SpMatCSR::InnerIterator it(*T, i); it; ++it)
      gsl_vector_set(rowSum, i, gsl_vector_get(rowSum, i)+it.value()); 
  
  return;
}

/** 
 * \brief Get the sum of each row of an Eigen CSR matrix of integer type.
 *
 * Get the sum of each row of an Eigen CSR matrix of integer type as a GSL vector.
 * \param[in] T    Eigen CSR matrix of integer type on which to sum.
 * \return Integer GSL vector of the sum of rows of the sparse matrix.
 */
gsl_vector_int *
getRowSum(SpMatIntCSR *T)
{
  int N = T->rows();
  gsl_vector_int *rowSum = gsl_vector_int_calloc(N);
  // Calculate row sums
  for (int i = 0; i < N; i++)
    for (SpMatIntCSR::InnerIterator it(*T, i); it; ++it)
      gsl_vector_int_set(rowSum, i, gsl_vector_int_get(rowSum, i)+it.value()); 
  
  return rowSum;
}

/** 
 * \brief Get the sum of each column of an Eigen CSR matrix.
 *
 * Get the sum of each column of an Eigen CSR matrix as a GSL vector.
 * \param[in] T    Eigen CSR matrix on which to sum.
 * \return GSL vector of the sum of columns of the sparse matrix.
 */
gsl_vector *
getColSum(SpMatCSR *T)
{
  int N = T->rows();
  gsl_vector *colSum = gsl_vector_calloc(N);

  // Calculate col sums
  for (int irow = 0; irow < N; irow++)
    for (SpMatCSR::InnerIterator it(*T, irow); it; ++it)
      gsl_vector_set(colSum, it.col(),
		     gsl_vector_get(colSum, it.col()) + it.value()); 
  
  return colSum;
}

/** 
 * \brief Get the sum of each column of an Eigen CSR matrix.
 *
 * Get the sum of each column of an Eigen CSR matrix as a GSL vector
 * (alternate version with output in argument).
 * \param[in] T    Eigen CSR matrix on which to sum.
 * \param[out] colSum GSL vector of the sum of columns of the sparse matrix.
 */
void
getColSum(SpMatCSR *T, gsl_vector *colSum)
{
  int N = T->rows();

  // Calculate col sums
  gsl_vector_set_all(colSum, 0.);
  for (int irow = 0; irow < N; irow++)
    for (SpMatCSR::InnerIterator it(*T, irow); it; ++it)
      gsl_vector_set(colSum, it.col(),
		     gsl_vector_get(colSum, it.col()) + it.value()); 
  
  return;
}

/** 
 * \brief Get the sum of each column of an Eigen CSC matrix.
 *
 * Get the sum of each column of an Eigen CSC matrix as a GSL vector.
 * \param[in] T    Eigen CSC matrix on which to sum.
 * \return GSL vector of the sum of columns of the sparse matrix.
 */
gsl_vector *
getColSum(SpMatCSC *T)
{
  int N = T->rows();
  gsl_vector *colSum = gsl_vector_calloc(N);
  // Calculate col sums
  for (int icol = 0; icol < N; icol++)
    for (SpMatCSC::InnerIterator it(*T, icol); it; ++it)
      gsl_vector_set(colSum, icol, gsl_vector_get(colSum, icol) + it.value()); 
  
  return colSum;
}

/** 
 * \brief Returns a the sum of the elements of an Eigen CSR matrix.
 *
 * Returns a the sum over all the elements of an Eigen CSR matrix.
 * \param[in] T    Eigen CSR matrix on which to sum.
 * \return Sum over all the elements of the sparse matrix.
 */
double
getSum(SpMatCSR *T)
{
  int N = T->rows();
  double sum = 0.;
  // Calculate col sums
  for (int irow = 0; irow < N; irow++)
    for (SpMatCSR::InnerIterator it(*T, irow); it; ++it)
      sum += it.value();
  
  return sum;
}


/** 
 * \brief Returns a the sum of the elements of an GSL vector.
 *
 * Returns a the sum over all the elements of an GSL vector.
 * \param[in] v    GSL vector over which to sum.
 * \return Sum over all the elements of the vector.
 */
double
sumVectorElements(gsl_vector *v)
{
  double sum = 0.;
  
  for (size_t i = 0; i < v->size; i++)
    sum += gsl_vector_get(v, i);
  
  return sum;
}

/** 
 * \brief Normalize a GSL vector by the sum of its elements.
 *
 * Normalize a GSL vector by the sum of its elements.
 * \param[in] v    GSL vector to normalize.
 */
void
normalizeVector(gsl_vector *v)
{
  double sum;

  sum = sumVectorElements(v);
  gsl_vector_scale(v, 1. / sum);
  
  return;
}

/** 
 * \brief Normalize each row of an Eigen CSR matrix by a GSL vector.
 *
 * Normalize each row of an Eigen CSR matrix by each element of a GSL vector.
 * \param[in] T    Eigen CSR matrix to normalize.
 * \param[in] rowSum    GSL vector used to normalize the rows of the sparse matrix.
 */
void
normalizeRows(SpMatCSR *T, gsl_vector *rowSum)
{
  double rowSumi;
  for (int i = 0; i < T->rows(); i++){
    rowSumi = gsl_vector_get(rowSum, i);
    for (SpMatCSR::InnerIterator it(*T, i); it; ++it)
      if (rowSumi > 0.)
	it.valueRef() /= rowSumi;
  }
  return ;
}

/** 
 * \brief Perform an element-wise greater than test on an Eigen CSC matrix.
 *
 * Perform an element-wise greater than test on an Eigen CSC matrix.
 * \param[in] T    Eigen CSC matrix to test.
 * \param[in] ref    Scalar against which to compare.
 * \return    Eigen CSC matrix of boolean type resulting from the test.
 */
SpMatCSCBool *
cwiseGT(SpMatCSC *T, double ref)
{
  int j;
  SpMatCSCBool *cmpT = new SpMatCSCBool(T->rows(), T->cols());
  for (j = 0; j < T->cols(); j++)
    for (SpMatCSC::InnerIterator it(*T, j); it; ++it)
      cmpT->insert(it.row(), j) = it.value() > ref;

  return cmpT;
}
  
/** 
 * \brief Perform an element-wise lower than test on an Eigen CSC matrix.
 * 
 * Perform an element-wise lower than test on an Eigen CSC matrix.
 * \param[in] T    Eigen CSC matrix to test.
 * \param[in] ref    Scalar against which to compare.
 * \return    Eigen CSC matrix of boolean type resulting from the test.
 */
SpMatCSCBool *
cwiseLT(SpMatCSC *T, double ref)
{
  int j;
  SpMatCSCBool *cmpT = new SpMatCSCBool(T->rows(), T->cols());
  for (j = 0; j < T->cols(); j++)
    for (SpMatCSC::InnerIterator it(*T, j); it; ++it)
      cmpT->insert(it.row(), j) = it.value() < ref;

  return cmpT;
}

/** 
 * \brief Perform an any operation on an Eigen CSC matrix of boolean type.
 *
 * Perform an any operation on an Eigen CSC matrix of boolean type.
 * \param[in] T    Eigen CSC matrix of boolean type on which to any.
 * \return    True if any, False otherwise.
 */
bool
any(SpMatCSCBool *T)
{
  int j;
  for (j = 0; j < T->cols(); j++)
    for (SpMatCSCBool::InnerIterator it(*T, j); it; ++it)
      if (it.value())
	return true;

  return false;
}

/** 
 * \brief Finds the maximum element of an Eigen CSC matrix.
 *
 * Finds the maximum element of an Eigen CSC matrix.
 * \param[in] T    Eigen CSC matrix from which to find the maximum.
 * \return    Value of the maximum.
 */
double
max(SpMatCSC *T)
{
  int j;
  SpMatCSC::InnerIterator it(*T, 0);
  double maxValue = it.value();
  for (j = 0; j < T->cols(); j++)
    for (SpMatCSC::InnerIterator it(*T, j); it; ++it)
      if (it.value() > maxValue)
	maxValue = it.value();

  return maxValue;
}


/** 
 * \brief Finds the minimum element of an Eigen CSC matrix.
 *
 * Finds the minimum element of an Eigen CSC matrix.
 * \param[in] T    Eigen CSC matrix from which to find the minimum.
 * \return    Value of the minimum.
 */
double
min(SpMatCSC *T)
{
  int j;
  SpMatCSC::InnerIterator it(*T, 0);
  double minValue = it.value();
  for (j = 0; j < T->cols(); j++)
    for (SpMatCSC::InnerIterator it(*T, j); it; ++it)
      if (it.value() < minValue)
	minValue = it.value();

  return minValue;
}


/** 
 * \brief Finds the position of the maximum element of an Eigen CSC matrix.
 *
 * Finds the position of the maximum element of an Eigen CSC matrix.
 * \param[in] T    Eigen CSC matrix from which to find the maximum.
 * \return    Vector of the indices of the maximum of the matrix.
 */
std::vector<int> *
argmax(SpMatCSC *T)
{
  int j;
  std::vector<int> *argmax = new std::vector<int>(2);
  SpMatCSC::InnerIterator it(*T, 0);
  double maxValue = it.value();
  for (j = 0; j < T->cols(); j++){
    for (SpMatCSC::InnerIterator it(*T, j); it; ++it){
      if (it.value() > maxValue){
	argmax->at(0) = it.row();
	argmax->at(1) = it.col();
	maxValue = it.value();
      }
    }
  }

  return argmax;
}

/** 
 * \brief Finds the position of the minimum element of an Eigen CSC matrix.
 *
 * Finds the position of the minimum element of an Eigen CSC matrix.
 * \param[in] T    Eigen CSC matrix from which to find the minimum.
 * \return    Vector of the indices of the minimum of the matrix.
 */
std::vector<int> *
argmin(SpMatCSC *T)
{
  int j;
  std::vector<int> *argmin = new std::vector<int>(2);
  SpMatCSC::InnerIterator it(*T, 0);
  double minValue = it.value();
  for (j = 0; j < T->cols(); j++){
    for (SpMatCSC::InnerIterator it(*T, j); it; ++it){
      if (it.value() < minValue){
	argmin->at(0) = it.row();
	argmin->at(1) = it.col();
	minValue = it.value();
      }
    }
  }

  return argmin;
}


#endif
