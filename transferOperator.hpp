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
#include <ATSuite/atio.hpp>
#if defined (WITH_OMP) && WITH_OMP == 1
#include <omp.h>
#endif

/** \file transferOperator.hpp
 * \brief Calculate discretized approximation of transfer operators from time series.
 *   
 * Calculate discretized approximation of transfer operators from time series.
 * The result is given as forward and backward Markov transition matrices and their distributions.
 */


/* 
 * Typedef declaration
 */

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


/*
 * Class declarations
 */

/** \brief Grid class.
 *
 * Grid class used for the Galerin approximation of transfer operators
 * by a transition matrix on a grid.
 */
class Grid {
  /** \brief Allocate memory. */
  void allocate(gsl_vector_uint *);
  /** \brief Get uniform rectangular. */
  void getRectGrid(gsl_vector_uint *, const gsl_vector *, const gsl_vector *);

public:
  /** Number of dimensions */
  size_t dim;
  /** Number of grid boxes */
  size_t N;
  /** Number of grid boxes per dimension */
  gsl_vector_uint *nx;
  /** Grid box bounds for each dimension */
  std::vector<gsl_vector *> *gridBounds;

  /** \brief Default constructor. */
  Grid(){}
  /** \brief Constructor allocating an empty grid. */
  Grid(gsl_vector_uint *nx_){ allocate(nx_); }
  /** \brief Construct a uniform rectangular grid with different dimensions. */
  Grid(gsl_vector_uint *, const gsl_vector *, const gsl_vector *);
  /** \brief Construct a uniform rectangular grid with same dimensions. */
  Grid(size_t, size_t, double, double);
  /** \brief Destructor. */
  ~Grid();
  
  /** \brief Print the grid to file. */
  int printGrid(const char *, const char *, bool);
};

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
  transferOperator(const gsl_matrix *, const gsl_matrix *, const Grid *);
  /** \brief Constructor from a long trajectory for a given grid and lag */
  transferOperator(const gsl_matrix *, const Grid *, size_t);
  /** \brief Destructor */
  ~transferOperator();

  // Output methods
  /** \brief Print forward transition matrix to file in compressed matrix format.*/
  int printForwardTransition(const char *, const char *);
  /** \brief Print backward transition matrix to file in compressed matrix format.*/
  int printBackwardTransition(const char *, const char *);
  /** \brief Print initial distribution to file.*/
  int printInitDist(const char *, const char *);
  /** \brief Print final distribution to file.*/
  int printFinalDist(const char *, const char *);
};


/*
 *  Functions declarations
 */

/** \brief Get membership matrix from initial and final states for a grid. */
gsl_matrix_uint *getGridMemMatrix(const gsl_matrix *, const gsl_matrix *, const Grid *);
/** \brief Get the grid membership vector from a single long trajectory. */
gsl_matrix_uint *getGridMemMatrix(const gsl_matrix *, const Grid *, const size_t);
/** \brief Get the grid membership matrix from a single long trajectory. */
gsl_vector_uint *getGridMemVector(const gsl_matrix *, const Grid *);
/** \brief Get the grid membership matrix from the membership vector for a given lag. */
gsl_matrix_uint *memVector2memMatrix(const gsl_vector_uint *, size_t);
/** \brief Concatenate a list of membership vectors into one membership matrix. */
gsl_matrix_uint * memVectorList2memMatrix(const std::vector<gsl_vector_uint *> *, size_t);
/** \brief Get membership to a grid box of a single realization. */
int getBoxMembership(const gsl_vector *, const Grid *);
/** \brief Get triplet vector from membership matrix. */
tripletUIntVector *getTransitionCountTriplet(const gsl_matrix_uint *, size_t);


/*
 * Constructors and destructors definitions
 */

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
 * \param[in] grid           Pointer to Grid object.
 */
transferOperator::transferOperator(const gsl_matrix *initStates,
				   const gsl_matrix *finalStates,
				   const Grid *grid)
{
  gsl_matrix_uint *gridMem;

  // Allocate
  N = grid->N;
  allocate(N);

  // Get grid membership matrix
  gridMem = getGridMemMatrix(initStates, finalStates, grid);

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
 * \param[in] grid           Pointer to Grid object.
 * \param[in] tauStep        Lag used to calculate the transitions.
 */
transferOperator::transferOperator(const gsl_matrix *states, const Grid *grid,
				   const size_t tauStep)
{
  gsl_matrix_uint *gridMem;

  // Allocate
  N = grid->N;
  allocate(N);

  // Get grid membership matrix from a single long trajectory
  gridMem = getGridMemMatrix(states, grid, tauStep);

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
 * Construct a uniform rectangular grid with specific bounds for each dimension.
 * \param[in] nx_        GSL vector giving the number of boxes for each dimension.
 * \param[in] xmin       GSL vector giving the minimum box limit for each dimension.
 * \param[in] xmax       GSL vector giving the maximum box limit for each dimension.
 */
Grid::Grid(gsl_vector_uint *nx_, const gsl_vector *xmin, const gsl_vector *xmax)
{
  // Allocate and build uniform rectangular grid
  getRectGrid(nx_, xmin, xmax);
}

/**
 * Construct a uniform rectangular grid with same bounds for each dimension.
 * \param[in] dim_        Number of dimensions.
 * \param[in] inx         Number of boxes, identically for each dimension.
 * \param[in] dxmin       Minimum box limit, identically for each dimension.
 * \param[in] dxmax       Maximum box limit, identically for each dimension.
 */
Grid::Grid(size_t dim_, size_t inx, double dxmin, double dxmax)
{
  // Convert to uniform vectors to call getRectGrid.
  gsl_vector_uint *nx_ = gsl_vector_uint_alloc(dim_);
  gsl_vector *xmin_ = gsl_vector_alloc(dim_);
  gsl_vector *xmax_ = gsl_vector_alloc(dim_);
  gsl_vector_uint_set_all(nx_, inx);
  gsl_vector_set_all(xmin_, dxmin);
  gsl_vector_set_all(xmax_, dxmax);

  // Allocate and build uniform rectangular grid
  getRectGrid(nx_, xmin_, xmax_);

  // Free
  gsl_vector_uint_free(nx_);
  gsl_vector_free(xmin_);
  gsl_vector_free(xmax_);
}

/** Destructor desallocates memory used by the grid. */
Grid::~Grid()
{
  gsl_vector_uint_free(nx);
  for (size_t d = 0; d < dim; d++)
    gsl_vector_free((*gridBounds)[d]);
  delete gridBounds;
}


/*
 * Methods definitions
 */

/**
 * Allocate memory for the transition matrices and distributions.
 * \param[in] gridSize Number of grid boxes.
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
 * Print forward transition matrix to file in compressed matrix format.
 * \param[in] path Path to the file in which to print.
 * \param[in] dataFormat      Format in which to print each element.
 * \return         Status.
 */
int
transferOperator::printForwardTransition(const char *path, const char *dataFormat="%lf")
{
  FILE *fp;

  // Open file
  if ((fp = fopen(path, "w")) == NULL){
    fprintf(stderr, "Can't open %s for printing the forward transition matrix!\n", path);
    return(EXIT_FAILURE);
  }

  // Print
  Eigen2Compressed(fp, P, dataFormat);

  // Close
  fclose(fp);

  return 0;
}

/**
 * Print backward transition matrix to file in compressed matrix format.
 * \param[in] path Path to the file in which to print.
 * \param[in] dataFormat      Format in which to print each element.
 * \return         Status.
 */
int
transferOperator::printBackwardTransition(const char *path, const char *dataFormat="%lf")
{
  FILE *fp;

  // Open file
  if ((fp = fopen(path, "w")) == NULL){
    fprintf(stderr, "Can't open %s for printing the backward transition matrix!\n", path);
    return(EXIT_FAILURE);
  }

  // Print
  Eigen2Compressed(fp, Q, dataFormat);

  // Close
  fclose(fp);

  return 0;
}

/**
 * Print initial distribution to file.
 * \param[in] path Path to the file in which to print.
 * \param[in] dataFormat      Format in which to print each element.
 * \return         Status.
 */
int
transferOperator::printInitDist(const char *path, const char *dataFormat="%lf")
{
  FILE *fp;

  // Open file
  if ((fp = fopen(path, "w")) == NULL){
    fprintf(stderr, "Can't open %s for printing the initial distribution!\n", path);
    return(EXIT_FAILURE);
  }

  // Print
  gsl_vector_fprintf(fp, initDist, dataFormat);

  // Close
  fclose(fp);

  return 0;
}

/**
 * Print final distribution to file.
 * \param[in] path Path to the file in which to print.
 * \param[in] dataFormat      Format in which to print each element.
 * \return         Status.
 */
int
transferOperator::printFinalDist(const char *path, const char *dataFormat="%lf")
{
  FILE *fp;

  // Open file
  if ((fp = fopen(path, "w")) == NULL){
    fprintf(stderr, "Can't open %s for printing the final distribution!\n", path);
    return(EXIT_FAILURE);
  }

  // Print
  gsl_vector_fprintf(fp, finalDist, dataFormat);

  // Close
  fclose(fp);

  return 0;
}

/**
 * Allocate memory for the grid.
 * \param[in] GSL vector of unsigned integers giving the number of boxes per dimension.
 */
void
Grid::allocate(gsl_vector_uint *nx_)
{
  dim = nx_->size;
  
  nx = gsl_vector_uint_alloc(dim);
  gsl_vector_uint_memcpy(nx, nx_);
  
  N = 1;
  gridBounds = new std::vector<gsl_vector *>(dim);
  for (size_t d = 0; d < dim; d++){
    N *= gsl_vector_uint_get(nx, d);
    (*gridBounds)[d] = gsl_vector_alloc(gsl_vector_uint_get(nx, d) + 1);
  }

  return;
}

/**
 * Get a uniform rectangular grid with specific bounds for each dimension.
 * \param[in] nx         GSL vector giving the number of boxes for each dimension.
 * \param[in] xmin       GSL vector giving the minimum box limit for each dimension.
 * \param[in] xmax       GSL vector giving the maximum box limit for each dimension.
 */
void
Grid::getRectGrid(gsl_vector_uint *nx_, const gsl_vector *xmin, const gsl_vector *xmax)
{
  double delta;
  
  // Allocate
  allocate(nx_);

  // Build uniform grid bounds
  for (size_t d = 0; d < dim; d++) {
    // Get spatial step
    delta = (gsl_vector_get(xmax, d) - gsl_vector_get(xmin, d))
      / gsl_vector_uint_get(nx, d);
    // Set grid bounds
    gsl_vector_set((*gridBounds)[d], 0, gsl_vector_get(xmin, d));
    for (size_t i = 1; i < gsl_vector_uint_get(nx, d) + 1; i++)
      gsl_vector_set((*gridBounds)[d], i,
		     gsl_vector_get((*gridBounds)[d], i-1) + delta);
  }

  return;
}

/**
 * Print the grid to file.
 * \param[in] path       Path to the file in which to print.
 * \param[in] dataFormat Format in which to print each element.
 * \param[in] verbose    If true, also print to the standard output.  
 * \return               Status.
 */
int
Grid::printGrid(const char *path, const char *dataFormat="%lf", bool verbose=false)
{
  gsl_vector *bounds;
  FILE *fp;

  // Open file
  if ((fp = fopen(path, "w")) == NULL){
    fprintf(stderr, "Can't open %s for printing the grid!\n", path);
    return(EXIT_FAILURE);
  }

  if (verbose)
    std::cout << "Domain grid (min, max, n):" << std::endl;

  // Print grid
  for (size_t d = 0; d < dim; d++) {
    bounds = (*gridBounds)[d];
    if (verbose) {
      std::cout << "dim " << d+1 << ": ("
		<< gsl_vector_get(bounds, 0) << ", "
		<< gsl_vector_get(bounds, bounds->size - 1) << ", "
		<< (bounds->size - 1) << ")" << std::endl;
    }
    
    for (size_t i = 0; i < bounds->size; i++){
      fprintf(fp, dataFormat, gsl_vector_get((*gridBounds)[d], i));
      fprintf(fp, " ");
    }
    fprintf(fp, "\n");
  }

  // Close
  fclose(fp);

  return 0;
}


/*
 * Function definitions
 */

/**
 * Get membership matrix from initial and final states for a grid.
 * \param[in] initStates     GSL matrix of initial states.
 * \param[in] finalStates    GSL matrix of final states.
 * \param[in] grid           Pointer to Grid object.
 * \return                   GSL grid membership matrix.
 */
gsl_matrix_uint *
getGridMemMatrix(const gsl_matrix *initStates, const gsl_matrix *finalStates,
		 const Grid *grid)
{
  const size_t nTraj = initStates->size1;
  const size_t dim = initStates->size2;
  gsl_matrix_uint *gridMem;

  // Allocate
  gridMem = gsl_matrix_uint_alloc(grid->N, 2);
  
  // Assign a pair of source and destination boxes to each trajectory
#pragma omp parallel
  {
    gsl_vector *X = gsl_vector_alloc(dim);
    
#pragma omp for
    for (size_t traj = 0; traj < nTraj; traj++) {
      // Find initial box
      gsl_matrix_get_row(X, initStates, traj);
      gsl_matrix_uint_set(gridMem, traj, 0, getBoxMembership(X, grid));
      
      // Find final box
      gsl_matrix_get_row(X, finalStates, traj);
      gsl_matrix_uint_set(gridMem, traj, 1, getBoxMembership(X, grid));
    }
    gsl_vector_free(X);
  }
  
  return gridMem;
}

/**
 * Get the grid membership vector from a single long trajectory.
 * \param[in] states         GSL matrix of states for each time step.
 * \param[in] grid           Pointer to Grid object.
 * \return                   GSL grid membership vector.
 */
gsl_vector_uint *
getGridMemVector(const gsl_matrix *states, const Grid *grid)
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
	gsl_vector_uint_set(gridMem, traj, getBoxMembership(X, grid));
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
memVector2memMatrix(gsl_vector_uint *gridMemVect, size_t tauStep)
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
 * \param[in] grid           Pointer to Grid object.
 * \param[in] tauStep        Lag used to calculate the transitions.
 * \return                   GSL grid membership matrix.
 */
gsl_matrix_uint *
getGridMemMatrix(const gsl_matrix *states, const Grid *grid, const size_t tauStep)
{
  // Get membership vector
  gsl_vector_uint *gridMemVect = getGridMemVector(states, grid);

  // Get membership matrix from vector
  gsl_matrix_uint *gridMem = memVector2memMatrix(gridMemVect, tauStep);

  // Free
  gsl_vector_uint_free(gridMemVect);
  
  return gridMem;
}

/**
 * Concatenate a list of membership vectors into one membership matrix.
 * \param[in] memList    STD vector of membership GSL vectors each of them associated
 * with a single long trajectory.
 * \param[in] tauStep    Lag used to calculate the transitions.
 * \return               GSL grid membership matrix.
 */
gsl_matrix_uint *
memVectorList2memMatrix(const std::vector<gsl_vector_uint *> *memList, size_t tauStep)
{
  size_t nStatesTot = 0;
  size_t count;
  const size_t listSize = memList->size();
  gsl_matrix_uint *gridMem, *gridMemMatrixL;

  // Get total number of states and allocate grid membership matrix
  for (size_t l = 0; l < listSize; l++)
    nStatesTot += (memList->at(l))->size;
  gridMem = gsl_matrix_uint_alloc(nStatesTot - tauStep * listSize, 2);
  
  // Get membership matrix from list of membership vectors
  count = 0;
  for (size_t l = 0; l < listSize; l++) {
    gridMemMatrixL = memVector2memMatrix(memList->at(l), tauStep);
    for (size_t t = 0; t < gridMemMatrixL->size1; t++) {
      gsl_matrix_uint_set(gridMem, count, 0,
			  gsl_matrix_uint_get(gridMemMatrixL, t, 0));
      gsl_matrix_uint_set(gridMem, count, 1,
			  gsl_matrix_uint_get(gridMemMatrixL, t, 1));
      count++;
    }
    gsl_matrix_uint_free(gridMemMatrixL);
  }
  
  return gridMem;
}

/**
 * Get membership to a grid box of a single realization.
 * \param[in] state          GSL vector of a single state.
 * \param[in] grid           Pointer to Grid object.
 * \return                   Box index to which the state belongs.
 */
int
getBoxMembership(const gsl_vector *state, const Grid *grid)
{
  const size_t dim = state->size;
  size_t inBox, nBoxDir;
  size_t foundBox;
  size_t subbp, subbn, ids;
  gsl_vector *bounds;

  // Get box
  foundBox = grid->N;
  for (size_t box = 0; box < grid->N; box++){
    inBox = 0;
    subbp = box;
    for (size_t d = 0; d < dim; d++){
      bounds = grid->gridBounds->at(d);
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
  tripletUIntVector *T = new tripletUIntVector;
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

#endif
