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

// Class declarations
/** \brief Transfer operator class.
 * 
 * Transfer operator class including
 * the forward and backward transition matrices
 * and the initial and final distributions calculated from data.
 * The constructors are based on membership matrices 
 * with the first column giving the box to which belong the initial state of trajectories
 * and the second column the box to which belong the final state of trajectories.
 * The initial and final states can also be directly given as to matrices
 * with each row giving a state.
 * Finally, transferOperator can also be constructed 
 * from a single long trajectory and a given lag.
 * Then, a membership vector is first calculated,
 * assigning each realization to a box,
 * from which the membership matrix can be calculated
 * for the lag given.
 */
class transferOperator {
  /** \brief Allocate memory. */
  void allocate(size_t);
/** \brief Get the transition matrices from a grid membership matrix. */
  void buildFromMembership(const gsl_matrix_uint *);
    
public:
  size_t N;              //!< Size of the grid
  SpMatCSR *P;           //!< Forward transition matrix
  SpMatCSR *Q;           //!< Backward transition matrix
  gsl_vector *initDist;  //!< Initial distribution
  gsl_vector *finalDist; //!< Final distribution

  /** \brief Default constructor */
  transferOperator(){}
  /** \brief Empty constructor allocating for grid size*/
  transferOperator(size_t gridSize){ allocate(gridSize); }
  /** \brief Constructor from the membership matrix. */
  transferOperator(const gsl_matrix_uint *, size_t);
  /** \brief Constructor from membership matrix and lag */
  transferOperator(const gsl_vector_uint *, size_t); 
  /** \brief Constructor from initial and final states for a given grid */
  transferOperator(const gsl_matrix *, const gsl_matrix *, const std::vector<gsl_vector *> *);
  /** \brief Constructor from a long trajectory for a given grid and lag */
  transferOperator(const gsl_matrix *, const std::vector<gsl_vector *> *, size_t);
  /** \brief Destructor */
  ~transferOperator();
};

// Functions declarations

/** \brief Get membership matrix from initial and final states for a grid. */
gsl_matrix_uint *getGridMembership(const gsl_matrix *, const gsl_matrix *,
				   const std::vector<gsl_vector *> *);
/** \brief Get the grid membership vector from a single long trajectory. */
gsl_matrix_uint *getGridMembership(const gsl_matrix *,
				   const std::vector<gsl_vector *> *, const size_t);
/** \brief Get the grid membership matrix from a single long trajectory. */
gsl_vector_uint *getGridMembership(const gsl_matrix *, const std::vector<gsl_vector *> *);
/** \brief Get the grid membership matrix from the membership vector for a given lag. */
gsl_matrix_uint *memVector2memMatrix(const gsl_vector_uint *, const size_t);
/** \brief Get membership to a grid box of a single realization. */
int getBoxMembership(const gsl_vector *, const std::vector<gsl_vector *> *);
/** \brief Get triplet vector from membership matrix. */
tripletUIntVector *getTransitionCountTriplet(const gsl_matrix_uint *, size_t);
/** \brief Get a uniform rectangular grid. */
std::vector<gsl_vector *> *getGridRect(size_t, size_t, double, double);
/** \brief Get a uniform rectangular grid. */
std::vector<gsl_vector *> *getGridRect(const gsl_vector_uint *,
				       const gsl_vector *, const gsl_vector *);
/** \brief Print a uniform rectangular grid to file. */
void writeGridRect(FILE *, const std::vector<gsl_vector *> *, bool);


// Definitions
/**
 * Allocate memory for the transition matrices and distributions.
 */
void
transferOperator::allocate(size_t gridSize)
{
  N = gridSize;
  P = new SpMatCSR(N, N);
  Q = new SpMatCSR(N, N);
  initDist = gsl_vector_alloc(N);
  finalDist = gsl_vector_alloc(N);
  
  return;
}

/**
 * Method called from the constructor to get the transition matrices
 * from a grid membership matrix.
 * \param[in] gridMem Grid membership matrix.
 */
void
transferOperator::buildFromMembership(const gsl_matrix_uint *gridMem)
{
  // Get transition count triplets
  tripletUIntVector *T = getTransitionCountTriplet(gridMem, N);
  
  // Convert to CSR matrix
  P->setFromTriplets(T->begin(), T->end());
  
  // Get initial and final distribution
  getRowSum(P, initDist);
  getColSum(P, finalDist);
  
  // Get forward and backward transition matrices
  *Q = SpMatCSR(P->transpose());
  toLeftStochastic(P);
  toLeftStochastic(Q);
  normalizeVector(initDist);
  normalizeVector(finalDist);

  // Free
  delete T;

  return;
}


/**
 * Construct transferOperator by calculating
 * the forward and backward transition matrices and distributions 
 * from the grid membership matrix.
 * \param[in] gridMem        GSL grid membership matrix.
 * \param[in] gridSize       Number of grid boxes.
 */
transferOperator::transferOperator(const gsl_matrix_uint *gridMem, size_t gridSize)
  {
  // Allocate
  allocate(gridSize);

  // Get transition matrices and distributions from grid membership matrix
  buildFromMembership(gridMem);

  return;
}

/**
 * Construct transferOperator by calculating
 * the forward and backward transition matrices and distributions 
 * from the initial and final states of trajectories.
 * \param[in] initStates     GSL matrix of initial states.
 * \param[in] finalStates    GSL matrix of final states.
 * \param[in] gridBounds     STD vector of gsl_vector of grid box bounds for each dimension.
 */
transferOperator::transferOperator(const gsl_matrix *initStates,
				   const gsl_matrix *finalStates,
				   const std::vector<gsl_vector *> *gridBounds)
{
  gsl_vector *bounds;
  gsl_matrix_uint *gridMem;

  // Get grid dimensions
  N = 1;
  for (size_t dir = 0; dir < initStates->size2; dir++){
    bounds = gridBounds->at(dir);
    N *= bounds->size - 1;
  }

  // Allocate
  allocate(N);

  // Get grid membership matrix
  gridMem = getGridMembership(initStates, finalStates, gridBounds);

  // Get transition matrices and distributions from grid membership matrix
  buildFromMembership(gridMem);

  // Free
  gsl_matrix_uint_free(gridMem);
  
  return;
}

/**
 * Construct transferOperator calculating the forward and backward transition matrices
 * and distributions from a single long trajectory, for a given grid and lag.
 * \param[in] states         GSL matrix of states for each time step.
 * \param[in] gridBounds     STD vector of gsl_vector of grid box bounds for each dimension.
 * \param[in] tauStep        Lag used to calculate the transitions.
 */
transferOperator::transferOperator(const gsl_matrix *states,
				   const std::vector<gsl_vector *> *gridBounds,
				   const size_t tauStep)
{
  gsl_vector *bounds;
  gsl_matrix_uint *gridMem;

  // Get grid dimensions
  N = 1;
  for (size_t dir = 0; dir < states->size2; dir++){
    bounds = gridBounds->at(dir);
    N *= bounds->size - 1;
  }

  // Allocate
  allocate(N);

  // Get grid membership matrix from a single long trajectory
  gridMem = getGridMembership(states, gridBounds, tauStep);

  // Get transition matrices and distributions from grid membership matrix
  buildFromMembership(gridMem);

  // Free
  gsl_matrix_uint_free(gridMem);
  
  return;
}

/** Destructor of transferOperator: desallocate all pointers. */
transferOperator::~transferOperator()
{
  delete P;
  delete Q;
  gsl_vector_free(initDist);
  gsl_vector_free(finalDist);
}

/**
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
 * Get the grid membership matrix from the membership vector for a given lag.
 * \param[in] gridMemVect    Grid membership vector of a long trajectory for a grid.
 * \param[in] tauStep        Lag used to calculate the transitions.
 * \return                   GSL grid membership matrix.
 */
gsl_matrix_uint *
memVector2memMatrix(gsl_vector_uint *gridMemVect, const size_t tauStep)
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
  gsl_vector_uint *gridMemVect;

  // Get membership vector
  gridMemVect = getGridMembership(states, gridBounds);

  // Get membership matrix from vector
  gridMem = memVector2memMatrix(gridMemVect, tauStep);

  // Free
  gsl_vector_uint_free(gridMemVect);
  
  return gridMem;
}

/**
 * Get membership to a grid box of a single realization.
 * \param[in] state          GSL vector of a single state.
 * \param[in] gridBounds     STD vector of gsl_vector of grid box bounds for each dimension.
 * \return                   Box index to which the state belongs.
 */
int
getBoxMembership(const gsl_vector *state, const std::vector<gsl_vector *> *gridBounds)
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
 * Get the triplet vector counting the transitions
 * from pairs of grid boxes from the grid membership matrix.
 * \param[in] gridMem Grid membership matrix.
 * \param[in] N       Size of the grid.
 * \return            Triplet vector counting the transitions.
 */
tripletUIntVector *
getTransitionCountTriplet(const gsl_matrix_uint *gridMem, size_t N)
{
  const size_t nTraj = gridMem->size1;
  size_t box0, boxf;
  size_t nOut = 0;
  tripletUIntVector *T = new tripletUintVector;
  T->reserve(nTraj);

  for (size_t traj = 0; traj < nTraj; traj++) {
    box0 = gsl_matrix_uint_get(gridMem, traj, 0);
    boxf = gsl_matrix_uint_get(gridMem, traj, 1);
    
    // Add transition triplet
    if ((box0 < N) && (boxf < N))
      T->push_back(tripletUInt(box0, boxf, 1));
    else
      nOut++;
  }
  std::cout <<  nOut * 100. / nTraj
	    << "% of the trajectories ended up out of the domain." << std::endl;

  return T;
}

/**
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
 * Get a uniform rectangular grid with specific bounds for each dimension.
 * \param[in] nx         GSL vector giving the number of boxes for each dimension.
 * \param[in] xmin       GSL vector giving the minimum box limit for each dimension.
 * \param[in] xmax       GSL vector giving the maximum box limit for each dimension.
 * \return               STD vector of gsl_vectors of grid box bounds for each dimension.
 */
std::vector<gsl_vector *> *
getGridRect(const gsl_vector_uint *nx,
	    const gsl_vector *xmin,
	    const gsl_vector *xmax)
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
 * Print a uniform rectangular grid to file.
 * \param[in] fp            File descriptor of the file to which to print the grid.
 * \param[in] gridBounds    STD vector of gsl_vector of grid box bounds for each dimension.
 * \param[in] verbose       If true, also print to the standard output.  
 */
void
writeGridRect(FILE *fp, const std::vector<gsl_vector *> *gridBounds,
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
