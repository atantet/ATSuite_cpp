#ifndef ATMARKOV_HPP
#define ATMARKOV_HPP

#include <list>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_vector_int.h>
#include <gsl/gsl_matrix.h>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <ATSuite/atmatrix.hpp>

/** \brief Calculates the log of x by x */
#define plog2p( x ) ( (x) > 0.0 ? (x) * log(x) / log(2) : 0.0 )

using namespace Eigen;

/** \file atmarkov.hpp
 *  \brief Markov transition matrix manipulation.
 *   
 *  ATSuite Markov transition matrix manipulation routines.
 */

// Typedef declaration
/** \brief Eigen CSR matrix of double type. */
typedef SparseMatrix<double, RowMajor> SpMatCSR;
/** \brief Eigen CSR matrix of integer type. */
typedef SparseMatrix<int, RowMajor> SpMatIntCSR;
/** \brief Eigen CSC matrix of double type. */
typedef SparseMatrix<double, ColMajor> SpMatCSC;
/** \brief Eigen CSC matrix of boolean type. */
typedef SparseMatrix<bool, ColMajor> SpMatCSCBool;

// Declarations
void toRightStochastic(SpMatCSC *);
void toRightStochastic(SpMatCSR *);
void toLeftStochastic(SpMatCSR *);
void toAndStochastic(SpMatCSR *);
void toAndStochastic(SpMatCSC *);
void filterTransitionMatrix(SpMatCSR *, gsl_vector *, gsl_vector *, double, int);
// double entropy(VectorXd *);
// double entropyRate(SpMatCSC *, VectorXd *);
// double entropyRate(MatrixXd *, VectorXd *);
// void condition4Entropy(SpMatCSC *);
// void lowlevelTransition(SpMatCSC *, VectorXd *, VectorXi *,
// 			MatrixXd *, VectorXd *);

// Definitions
/**
 * Make an Eigen CSC matrix right stochastic
 * by normalizing each row by the sum of its elements.
 * \param[in] T    Eigen CSC matrix to make stochastic.
 */
void toRightStochastic(SpMatCSC *T)
{
  int j;
  VectorXd rowSum = VectorXd::Zero(T->rows());
  // Calculate row sums
  for (j = 0; j < T->cols(); j++)
    for (SpMatCSC::InnerIterator it(*T, j); it; ++it)
      rowSum(it.row()) += it.value();

  // Normalize rows
  for (j = 0; j < T->cols(); j++)
    for (SpMatCSC::InnerIterator it(*T, j); it; ++it)
      if (rowSum(it.row()) > 0.)
	it.valueRef() /= rowSum(it.row());

  return;
}
  
/**
 * Make an Eigen CSR matrix right stochastic
 * by normalizing each row by the sum of its elements.
 * \param[in] T    Eigen CSR matrix to make stochastic.
 */
void toRightStochastic(SpMatCSR *T)
{
  // Get row sum vector
  gsl_vector *rowSum = getRowSum(T);

  // Normalize rows
  normalizeRows(T, rowSum);

  // Free row sum vector
  gsl_vector_free(rowSum);
  
  return;
}
  
/**
 * Make an Eigen CSR matrix left stochastic
 * by normalizing each column by the sum of its elements.
 * \param[in] T    Eigen CSR matrix to make stochastic.
 */
void toLeftStochastic(SpMatCSR *T)
{
  int j;
  VectorXd colSum = VectorXd::Zero(T->cols());
  // Calculate row sums
  for (j = 0; j < T->rows(); j++)
    for (SpMatCSR::InnerIterator it(*T, j); it; ++it)
      colSum(it.col()) += it.value();

  // Normalize rows
  for (j = 0; j < T->rows(); j++)
    for (SpMatCSR::InnerIterator it(*T, j); it; ++it)
      if (colSum(it.col()) > 0.)
	it.valueRef() /= colSum(it.col());

  return;
}

/**
 * Make an Eigen CSR matrix and stochastic
 * by normalizing it by the sum of its elements.
 * \param[in] T    Eigen CSR matrix to make stochastic.
 */
void toAndStochastic(SpMatCSR *T)
{
  double norm = getSum(T);
  // Normalize
  for (int outerIdx = 0; outerIdx < T->outerSize(); outerIdx++)
    for (SpMatCSR::InnerIterator it(*T, outerIdx); it; ++it)
      it.valueRef() /= norm;
  return;
}

/**
 * Make an Eigen CSC matrix and stochastic
 * by normalizing it by the sum of its elements.
 * \param[in] T    Eigen CSC matrix to make stochastic.
 */
void toAndStochastic(SpMatCSC *T)
{
  double norm = 0.;
  // Get Norm
  for (int outerIdx = 0; outerIdx < T->outerSize(); outerIdx++)
    for (SpMatCSC::InnerIterator it(*T, outerIdx); it; ++it)
      norm += it.value();
  // Normalize
  for (int outerIdx = 0; outerIdx < T->outerSize(); outerIdx++)
    for (SpMatCSC::InnerIterator it(*T, outerIdx); it; ++it)
      it.valueRef() /= norm;
  return;
}

/**
 * Remove weak nodes from a transition matrix
 * based on initial and final probability of each Markov state.
 * \param[inout] T    The Eigen CSR transition matrix to filter.
 * \param[in] rowCut  The probability distribution associated with each row.
 * \param[in] colCut  The probability distribution associated with each column.
 * \param[in] tol     Probability under which Markov states are removed.
 * \param[in] norm    Choice of normalization,
 * - norm = 0: normalize over all elements,
 * - norm = 1: to right stochastic,
 * - norm = 2: to left stochastic,
 * - no normalization for any other choice.
 */
void
filterTransitionMatrix(SpMatCSR *T,
		       gsl_vector *rowCut, gsl_vector *colCut,
		       double tol, int norm)
{
  double valRow, valCol;
  
  for (int outerIdx = 0; outerIdx < T->outerSize(); outerIdx++){
    valRow = gsl_vector_get(rowCut, outerIdx);
    for (SpMatCSR::InnerIterator it(*T, outerIdx); it; ++it){
      valCol = gsl_vector_get(colCut, it.col());
      // Remove elements of states to be removed
      if ((valRow < tol) || (valCol < tol))
	it.valueRef() = 0.;
    }
  }
  //T->prune();
    
  // Normalize
  switch (norm){
  case 0:
    toAndStochastic(T);
    break;
  case 1:
    toRightStochastic(T);
    break;
  case 2:
    toLeftStochastic(T);
    break;
  default:
    break;
  }
  normalizeVector(rowCut);
  normalizeVector(colCut);

  return;
}

// double entropy(VectorXd *dist)
// {
//   double s = 0.;
//   for (int k = 0; k < dist->size(); k++)
//     s -= plog2p((*dist)(k));
//   return s;
// }

// double entropyRate(SpMatCSC *T, VectorXd *dist)
// {
//   int i, j;
//   double s = 0.;
//   for (j = 0; j < T->outerSize(); j++){
//     for (SpMatCSC::InnerIterator it(*T, j); it; ++it){
//       i = it.row();
//       // Increment low-level transition matrix
//       s -= (*dist)(i) * plog2p(it.value());
//     }
//   }
//   return s;
// }

// double entropyRate(MatrixXd *T, VectorXd *dist)
// {
//   int i, j;
//   double s = 0.;
//   for(j = 0; j < T->cols(); j++){
//     for (i = 0; i < T->rows(); i++){
//       // Increment low-level transition matrix
//       s -= (*dist)(i) * plog2p((*T)(i, j));
//     }
//   }
//   return s;
// }  

// void condition4Entropy(SpMatCSC *T)
// {
//   int j;
//   VectorXd rowSum = VectorXd::Zero(T->rows());
//   // Calculate row sums
//   for (j = 0; j < T->cols(); j++){
//     for (SpMatCSC::InnerIterator it(*T, j); it; ++it){
//       if (it.value() > 1.)
// 	it.valueRef() = 1.;
//       if (it.value() < 0.)
// 	it.valueRef() = 0.;
//     }
//   }

//   return;
// }

// void lowlevelTransition(SpMatCSC *highT, VectorXd *highDist, VectorXi *member,
// 			MatrixXd *lowT, VectorXd *lowDist)
// {
//   const int N = highT->rows();
//   int ncom;
//   int i, j;

//   // Get set of communities from membership vector
//   list<int>  coms(member->data(), member->data() + N);
//   coms.sort();
//   coms.unique();
//   ncom = coms.size();

//   // Allocate and initialize low-level transition matrix
//   *lowT = MatrixXd::Zero(ncom, ncom);
//   *lowDist = VectorXd::Zero(ncom);

//   // Get community map
//   map<int, int> comMap;
//   list<int>::iterator it = coms.begin();
//   for (int k = 0; k < ncom; k++){
//     comMap[*it] = k;
//     advance(it, 1);
//   }

//   // Increment low-level stationary distribution
//   for (int i = 0; i < N; i++)
//     (*lowDist)(comMap[(*member)(i)]) += (*highDist)(i);

//   for (j = 0; j < N; j++){
//     for (SpMatCSC::InnerIterator it(*highT, j); it; ++it){
//       i = it.row();
//       // Increment low-level transition matrix
//       (*lowT)(comMap[(*member)(i)], comMap[(*member)(j)]) += (*highDist)(i) * it.value();
//     }
//   }
//   // Normalize
//   for (i = 0; i < ncom; i++)
//     for (j = 0; j < ncom; j++)
//       (*lowT)(i, j) = (*lowT)(i, j) / (*lowDist)(i);

//   return;
// }

#endif
