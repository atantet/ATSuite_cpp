#ifndef TRANSFEROPERATOR_HPP
#define TRANSFEROPERATOR_HPP

#include <iostream>
#include <vector>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_vector_uint.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_matrix_uint.h>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <ATSuite/atmatrix.hpp>
#include <ATSuite/atmarkov.hpp>
#if defined (WITH_OMP) && WITH_OMP == 1
#include <omp.h>
#endif
/** \file transferOperator.hpp
 * \brief Calculate discretized approximation of transfer operators from time series.
 *   
 * Calculate discretized approximation of transfer operators from time series.
 * The result is given as forward and backward Markov transition matrices and their distributions.
 */

// Typedef declaration
/** \brief Eigen triplet of double. */
typedef Eigen::Triplet<double> triplet;
/** \brief STD vector of Eigen triplet of double. */
typedef std::vector<triplet> tripletVector;
/** \brief Eigen triplet of integer. */
typedef Eigen::Triplet<size_t> tripletUInt;
/** \brief STD vector of Eigen triplet of integer. */
typedef std::vector<tripletUInt> tripletUIntVector;
/** \brief Eigen CSC matrix of double type. */
typedef Eigen::SparseMatrix<double, Eigen::ColMajor> SpMatCSC;
/** \brief Eigen CSR matrix of double type. */
typedef Eigen::SparseMatrix<double, Eigen::RowMajor> SpMatCSR;


// Decalrations
void getTransitionMatrix(const gsl_matrix_uint *, const size_t,
			 SpMatCSR *, SpMatCSR *, gsl_vector *, gsl_vector *);
void getTransitionMatrix(const gsl_matrix *, const gsl_matrix *,
			 const std::vector<gsl_vector *> *,
			 SpMatCSR *, SpMatCSR *, gsl_vector *, gsl_vector *);
void getTransitionMatrix(const gsl_matrix *, const std::vector<gsl_vector *> *,
			 const size_t tauStep,
			 SpMatCSR *, SpMatCSR *, gsl_vector *, gsl_vector *);
gsl_matrix_uint *getGridMembership(const gsl_matrix *, const gsl_matrix *,
				  const std::vector<gsl_vector *> *);
gsl_matrix_uint *getGridMembership(const gsl_matrix *,
				  const std::vector<gsl_vector *> *,
				  const size_t);
gsl_vector_uint *getGridMembership(const gsl_matrix *,
				  const std::vector<gsl_vector *> *);
gsl_matrix_uint *getGridMembership(gsl_vector_uint *, const size_t);
int getBoxMembership(gsl_vector *, const std::vector<gsl_vector *> *);
std::vector<gsl_vector *> *getGridRect(size_t, size_t, double, double);
std::vector<gsl_vector *> *getGridRect(gsl_vector_uint *,
				       gsl_vector *, gsl_vector *);
void writeGridRect(FILE *, std::vector<gsl_vector *> *, bool);


// Definitions
/**
 * \brief Get the transition matrices from the membership matrix.
 *
 * Get the forward and backward transition matrices and distributions from the membership matrix.
 * \param[in] gridMem        GSL grid membership matrix.
 * \param[in] N              Number of grid boxes.
 * \param[out] P             Eigen CSR forward transition matrix.
 * \param[out] Q             Eigen CSR backward transition matrix.
 * \param[out] initDist      GSL vector of initial distribution.
 * \param[out] finalDist     GSL vector of final distribution.
 */
void
getTransitionMatrix(const gsl_matrix_uint *gridMem, const size_t N,
		    SpMatCSR *P, SpMatCSR *Q,
		    gsl_vector *initDist, gsl_vector *finalDist)
{
  size_t nOut = 0;
  const size_t nTraj = gridMem->size1;
  size_t box0, boxf;
  tripletUIntVector T;
  T.reserve(nTraj);

  // Get transition count triplets
  for (size_t traj = 0; traj < nTraj; traj++) {
    box0 = gsl_matrix_uint_get(gridMem, traj, 0);
    boxf = gsl_matrix_uint_get(gridMem, traj, 1);
    
    // Add transition triplet
    if ((box0 < N) && (boxf < N))
      T.push_back(tripletUInt(box0, boxf, 1));
    else
      nOut++;
  }
  std::cout <<  nOut * 100. / nTraj
	    << "% of the trajectories ended up out of the domain." << std::endl;
  
  // Get correlation matrix
  P->setFromTriplets(T.begin(), T.end());
  // Get initial and final distribution
  getRowSum(P, initDist);
  getColSum(P, finalDist);
  // Get forward and backward transition matrices
  *Q = SpMatCSR(P->transpose());
  toLeftStochastic(P);
  toLeftStochastic(Q);
  normalizeVector(initDist);
  normalizeVector(finalDist);

  return;
}

/**
 * \brief Get the transition matrices from the initial and final states of trajectories.
 *
 * Get the forward and backward transition matrices and distributions
 * from the initial and final states of trajectories.
 * \param[in] initStates     GSL matrix of initial states.
 * \param[in] finalStates    GSL matrix of final states.
 * \param[in] gridBounds     STD vector of gsl_vector of grid box bounds for each dimension.
 * \param[out] P             Eigen CSR forward transition matrix.
 * \param[out] Q             Eigen CSR backward transition matrix.
 * \param[out] initDist      GSL vector of initial distribution.
 * \param[out] finalDist     GSL vector of final distribution.
 */
void
getTransitionMatrix(const gsl_matrix *initStates,
		    const gsl_matrix *finalStates,
		    const std::vector<gsl_vector *> *gridBounds,
		    SpMatCSR *P, SpMatCSR *Q,
		    gsl_vector *initDist, gsl_vector *finalDist)
{
  const size_t nTraj = initStates->size1;
  const size_t dim = initStates->size2;
  size_t N = 1;
  size_t nOut = 0;
  size_t box0, boxf;
  gsl_vector *bounds;
  gsl_matrix_uint *gridMem;
  tripletUIntVector T;

  // Get grid dimensions
  for (size_t dir = 0; dir < dim; dir++){
    bounds = gridBounds->at(dir);
    N *= bounds->size - 1;
  }
  T.reserve(nTraj);
  P = new SpMatCSR(N, N);
  Q = new SpMatCSR(N, N);

  // Get grid membership
  gridMem = getGridMembership(initStates, finalStates, gridBounds);
  
  for (size_t traj = 0; traj < nTraj; traj++) {
    box0 = gsl_matrix_uint_get(gridMem, traj, 0);
    boxf = gsl_matrix_uint_get(gridMem, traj, 1);
    
    // Add transition to matrix
    if ((box0 < N) && (boxf < N))
      T.push_back(tripletUInt(box0, boxf, 1));
    else
      nOut++;
  }
  std::cout <<  nOut * 100. / nTraj
	    << " of the trajectories ended up out of the domain." << std::endl;
  
  // Get correlation matrix
  P->setFromTriplets(T.begin(), T.end());
  // Get initial and final distribution
  initDist = getRowSum(P);
  finalDist = getColSum(P);
  // Get forward and backward transition matrices
  *Q = SpMatCSR(P->transpose());
  toLeftStochastic(P);
  toLeftStochastic(Q);
  normalizeVector(initDist);
  normalizeVector(finalDist);
   
  return;
}

/**
 * \brief Get membership matrix from initial and final states for a grid.
 *
 * Get membership matrix from initial and final states for a grid.
 * \param[in] initStates     GSL matrix of initial states.
 * \param[in] finalStates    GSL matrix of final states.
 * \param[in] gridBounds     STD vector of gsl_vector of grid box bounds for each dimension.
 * \return                   GSL grid membership matrix.
 */
gsl_matrix_uint *
getGridMembership(const gsl_matrix *initStates,
		  const gsl_matrix *finalStates,
		  const std::vector<gsl_vector *> *gridBounds)
{
  const size_t nTraj = initStates->size1;
  const size_t dim = initStates->size2;
  size_t N = 1;
  gsl_vector *bounds;
  gsl_matrix_uint *gridMem;

  // Get grid size
  for (size_t dir = 0; dir < dim; dir++){
    bounds = gridBounds->at(dir);
    N *= bounds->size - 1;
  }
  gridMem = gsl_matrix_uint_alloc(N, 2);
  
  // Assign a pair of source and destination boxes to each trajectory
#pragma omp parallel
  {
    gsl_vector *X = gsl_vector_alloc(dim);
    
#pragma omp for
    for (size_t traj = 0; traj < nTraj; traj++) {
      // Find initial box
      gsl_matrix_get_row(X, initStates, traj);
      gsl_matrix_uint_set(gridMem, traj, 0, getBoxMembership(X, gridBounds));
      
      // Find final box
      gsl_matrix_get_row(X, finalStates, traj);
      gsl_matrix_uint_set(gridMem, traj, 1, getBoxMembership(X, gridBounds));
    }
    gsl_vector_free(X);
  }
  
  return gridMem;
}

/**
 * \brief Get the transition matrices from a single long trajectory.
 *
 * Get the forward and backward transition matrices and distributions
 * from a single long trajectory.
 * \param[in] states         GSL matrix of states for each time step.
 * \param[in] gridBounds     STD vector of gsl_vector of grid box bounds for each dimension.
 * \param[in] tauStep        Lag used to calculate the transitions.
 * \param[out] P             Eigen CSR forward transition matrix.
 * \param[out] Q             Eigen CSR backward transition matrix.
 * \param[out] initDist      GSL vector of initial distribution.
 * \param[out] finalDist     GSL vector of final distribution.
 */
void
getTransitionMatrix(const gsl_matrix *states,
		    const std::vector<gsl_vector *> *gridBounds,
		    const size_t tauStep,
		    SpMatCSR *P, SpMatCSR *Q,
		    gsl_vector *initDist, gsl_vector *finalDist)
{
  size_t nTraj;
  const size_t dim = states->size2;
  size_t N = 1;
  size_t nOut = 0;
  size_t box0, boxf;
  gsl_vector *bounds;
  gsl_matrix_uint *gridMem;
  tripletUIntVector T;
  P = new SpMatCSR(N, N);
  Q = new SpMatCSR(N, N);

  // Get grid dimensions
  for (size_t dir = 0; dir < dim; dir++){
    bounds = gridBounds->at(dir);
    N *= bounds->size - 1;
  }

  // Get grid membership
  gridMem = getGridMembership(states, gridBounds, tauStep);
  nTraj = gridMem->size1;
  T.reserve(nTraj);
  
  for (size_t traj = 0; traj < nTraj; traj++) {
    box0 = gsl_matrix_uint_get(gridMem, traj, 0);
    boxf = gsl_matrix_uint_get(gridMem, traj, 1);
    
    // Add transition to matrix
    if ((box0 < N) && (boxf < N))
      T.push_back(tripletUInt(box0, boxf, 1));
    else
      nOut++;
  }
  std::cout <<  nOut * 100. / nTraj
	    << " of the trajectories ended up out of the domain." << std::endl;
  
  // Get correlation matrix
  P->setFromTriplets(T.begin(), T.end());
  // Get initial and final distribution
  initDist = getRowSum(P);
  finalDist = getColSum(P);
  // Get forward and backward transition matrices
  *Q = SpMatCSR(P->transpose());
  toLeftStochastic(P);
  toLeftStochastic(Q);
  normalizeVector(initDist);
  normalizeVector(finalDist);
   
  return;
}

/**
 * \brief Get the grid membership matrix from a single long trajectory.
 *
 * Get the grid membership matrix from a single long trajectory.
 * \param[in] states         GSL matrix of states for each time step.
 * \param[in] gridBounds     STD vector of gsl_vector of grid box bounds for each dimension.
 * \param[in] tauStep        Lag used to calculate the transitions.
 * \return                   GSL grid membership matrix.
 */
gsl_matrix_uint *
getGridMembership(const gsl_matrix *states,
		  const std::vector<gsl_vector *> *gridBounds,
		  const size_t tauStep)
{
  const size_t nStates = states->size1;
  gsl_vector_uint *gridMemVect;
  gsl_matrix_uint *gridMem = gsl_matrix_uint_alloc(nStates - tauStep, 2);

  // Get membership vector
  gridMemVect = getGridMembership(states, gridBounds);

  // Get membership matrix from vector
  for (size_t traj = 0; traj < (nStates - tauStep); traj++) {
    gsl_matrix_uint_set(gridMem, traj, 0,
		       gsl_vector_uint_get(gridMemVect, traj));
    gsl_matrix_uint_set(gridMem, traj, 1,
		       gsl_vector_uint_get(gridMemVect, traj + tauStep));
  }

  // Free
  gsl_vector_uint_free(gridMemVect);
  
  return gridMem;
}

/**
 * \brief Get the grid membership vector from a single long trajectory.
 *
 * Get the grid membership vector from a single long trajectory.
 * \param[in] states         GSL matrix of states for each time step.
 * \param[in] gridBounds     STD vector of gsl_vector of grid box bounds for each dimension.
 * \return                   GSL grid membership vector.
 */
gsl_vector_uint *
getGridMembership(const gsl_matrix *states,
		  const std::vector<gsl_vector *> *gridBounds)
{
  const size_t nStates = states->size1;
  const size_t dim = states->size2;
  gsl_vector_uint *gridMem = gsl_vector_uint_alloc(nStates);

  // Assign a pair of source and destination boxes to each trajectory
#pragma omp parallel
  {
    gsl_vector *X = gsl_vector_alloc(dim);
    
#pragma omp for
    for (size_t traj = 0; traj < nStates; traj++) {
      // Find initial box
      gsl_matrix_get_row(X, states, traj);
#pragma omp critical
      {
	gsl_vector_uint_set(gridMem, traj, getBoxMembership(X, gridBounds));
      }
    }
    gsl_vector_free(X);
  }
  
  return gridMem;
}

/**
 * \brief Get the grid membership matrix from the membership vector for a given lag.
 *
 * Get the grid membership matrix from the membership vector for a given lag.
 * \param[in] gridMemVect    Grid membership vector of a long trajectory for a grid.
 * \param[in] tauStep        Lag used to calculate the transitions.
 * \return                   GSL grid membership matrix.
 */
gsl_matrix_uint *
getGridMembership(gsl_vector_uint *gridMemVect,
		  const size_t tauStep)
{
  const size_t nStates = gridMemVect->size;
  gsl_matrix_uint *gridMem = gsl_matrix_uint_alloc(nStates - tauStep, 2);

  // Get membership matrix from vector
  for (size_t traj = 0; traj < (nStates - tauStep); traj++) {
    gsl_matrix_uint_set(gridMem, traj, 0,
		       gsl_vector_uint_get(gridMemVect, traj));
    gsl_matrix_uint_set(gridMem, traj, 1,
		       gsl_vector_uint_get(gridMemVect, traj + tauStep));
  }

  return gridMem;
}


/**
 * \brief Get membership to a grid box of a single realization.
 *
 * Get membership to a grid box of a single realization.
 * \param[in] state          GSL vector of a single state.
 * \param[in] gridBounds     STD vector of gsl_vector of grid box bounds for each dimension.
 * \return                   Box index to which the state belongs.
 */
int
getBoxMembership(gsl_vector *state, const std::vector<gsl_vector *> *gridBounds)
{
  const size_t dim = state->size;
  size_t inBox, nBoxDir;
  size_t foundBox;
  size_t subbp, subbn, ids;
  gsl_vector *bounds;
  size_t N = 1;

  // Get grid dimensions
  for (size_t d = 0; d < dim; d++){
    bounds = gridBounds->at(d);
    N *= bounds->size - 1;
  }

  // Get box
  foundBox = N;
  for (size_t box = 0; box < N; box++){
    inBox = 0;
    subbp = box;
    for (size_t d = 0; d < dim; d++){
      bounds = gridBounds->at(d);
      nBoxDir = bounds->size - 1;
      subbn = (size_t) (subbp / nBoxDir);
      ids = subbp - subbn * nBoxDir;
      inBox += (size_t) ((gsl_vector_get(state, d)
			  >= gsl_vector_get(bounds, ids))
			 & (gsl_vector_get(state, d)
			    < gsl_vector_get(bounds, ids+1)));
      subbp = subbn;
    }
    if (inBox == dim){
      foundBox = box;
      break;
    }
  }
  
  return foundBox;
}

/**
 * \brief Get a uniform rectangular grid.
 *
 * Get a uniform rectangular grid identically for each dimension.
 * \param[in] dim        Number of dimensions.
 * \param[in] nx         Number of boxes, identically for each dimension.
 * \param[in] xmin       Minimum box limit, identically for each dimension.
 * \param[in] xmax       Maximum box limit, identically for each dimension.
 * \return               STD vector of gsl_vectors of grid box bounds for each dimension.
 */
std::vector<gsl_vector *> *getGridRect(size_t dim, size_t nx,
				       double xmin, double xmax)
{
  double delta;
  std::vector<gsl_vector *> *gridBounds = new std::vector<gsl_vector *>(dim);

  for (size_t d = 0; d < dim; d++) {
    // Alloc one dimensional box boundaries vector
    (*gridBounds)[d] = gsl_vector_alloc(nx + 1);
    // Get spatial step
    delta = (xmax - xmin) / nx;
    gsl_vector_set((*gridBounds)[d], 0, xmin);
    
    for (size_t i = 1; i < nx + 1; i++)
      gsl_vector_set((*gridBounds)[d], i,
		     gsl_vector_get((*gridBounds)[d], i-1) + delta);
  }

  return gridBounds;
}

/**
 * \brief Get a uniform rectangular grid.
 * 
 * Get a uniform rectangular grid with specific bounds for each dimension.
 * \param[in] dim        Number of dimensions.
 * \param[in] nx         GSL vector giving the number of boxes for each dimension.
 * \param[in] xmin       GSL vector giving the minimum box limit for each dimension.
 * \param[in] xmax       GSL vector giving the maximum box limit for each dimension.
 * \return               STD vector of gsl_vectors of grid box bounds for each dimension.
 */
std::vector<gsl_vector *> *
getGridRect(gsl_vector_uint *nx, gsl_vector *xmin, gsl_vector *xmax)
{
  const size_t dim = nx->size;
  double delta;
  std::vector<gsl_vector *> *gridBounds = new std::vector<gsl_vector *>(dim);

  for (size_t d = 0; d < dim; d++) {
    // Alloc one dimensional box boundaries vector
    (*gridBounds)[d] = gsl_vector_alloc(gsl_vector_uint_get(nx, d) + 1);
    // Get spatial step
    delta = (gsl_vector_get(xmax, d) - gsl_vector_get(xmin, d))
      / gsl_vector_uint_get(nx, d);
    gsl_vector_set((*gridBounds)[d], 0, gsl_vector_get(xmin, d));
    
    for (size_t i = 1; i < gsl_vector_uint_get(nx, d) + 1; i++)
      gsl_vector_set((*gridBounds)[d], i,
		     gsl_vector_get((*gridBounds)[d], i-1) + delta);
  }

  return gridBounds;
}

/**
 * \brief Print a uniform rectangular grid to file.
 *
 * Print a uniform rectangular grid to file.
 * \param[in] fp            File descriptor of the file to which to print the grid.
 * \param[in] gridBounds    STD vector of gsl_vector of grid box bounds for each dimension.
 * \param[in] verbose       If true, also print to the standard output.  
 */
void
writeGridRect(FILE *fp, std::vector<gsl_vector *> *gridBounds,
	      bool verbose=false)
{
  gsl_vector *bounds;
  size_t dim = gridBounds->size();
  
  if (verbose)
    std::cout << "Domain grid (min, max, n):" << std::endl;
  
  for (size_t d = 0; d < dim; d++) {
    bounds = (*gridBounds)[d];
    if (verbose) {
      std::cout << "dim " << d+1 << ": ("
		<< gsl_vector_get(bounds, 0) << ", "
		<< gsl_vector_get(bounds, bounds->size - 1) << ", "
		<< (bounds->size - 1) << ")" << std::endl;
    }
    
    for (size_t i = 0; i < bounds->size; i++)
      fprintf(fp, "%lf ", gsl_vector_get((*gridBounds)[d], i));
    fprintf(fp, "\n");
  }

  return;
}


#endif
