#ifndef ATIO_HPP
#define ATIO_HPP

#include <vector>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>

/** \file atio.hpp
 *  \brief Input, output and conversion routines.
 *   
 *  ATSuite input, output and conversion routines between various matrices types.
 * 
 * Sparse matrices are printed in the following ascii compressed format:
 * 1. a header with the matrix type ("CSR" or "CSC"), inner size, outer size
 * and number of nonzero elements,
 * 2. the data vector,
 * 3. the indices vector,
 * 4. the pointer vector.
 */

// Typedef declaration
/** \brief Eigen sparse CSR matrix of double type. */
typedef Eigen::SparseMatrix<double, Eigen::RowMajor> SpMatCSR;
/** \brief Eigen sparse CSC matrix of double type. */
typedef Eigen::SparseMatrix<double, Eigen::ColMajor> SpMatCSC;
/** \brief Eigen triplet of double. */
typedef Eigen::Triplet<double> Tri;


// Declarations
void Eigen2Pajek(FILE *, SpMatCSR *);
void Eigen2Compressed(FILE *, SpMatCSC *);
void Eigen2Compressed(FILE *, SpMatCSR *);
SpMatCSC * pajek2Eigen(FILE *);
SpMatCSR * Compressed2Eigen(FILE *);
gsl_matrix * Compressed2EdgeList(FILE *);
void Compressed2EdgeList(FILE *, FILE *);
SpMatCSR * CSC2CSR(SpMatCSC *T);
SpMatCSC * CSR2CSC(SpMatCSR *T);
std::vector<Tri> Eigen2Triplet(SpMatCSC *);
std::vector<Tri> Eigen2Triplet(SpMatCSR *);
void fprintfEigen(FILE *, SpMatCSR *, const char *);
size_t lineCount(FILE *);

// Definitions
/**
 * \brief Print an Eigen CSR matrix in Pajek format.
 *
 * Print an Eigen CSR matrix in Pajek format
 * (see <a href="http://mrvar.fdv.uni-lj.si/pajek/">Pajek documentation</a>).
 * \param[in] fp    Descriptor of the file to which to print.
 * \param[in] P    Eigen matrix to print.
 */
void
Eigen2Pajek(FILE *fp, SpMatCSR *P){
  int N = P->rows();
  int E = P->nonZeros();

  // Write vertices
  fprintf(fp, "*Vertices %d\n", N);
  for (int k = 0; k < N; k++)
    fprintf(fp, "%d \"%d\"\n", k, k);

  // Write Edges
  fprintf(fp, "Edges %d\n", E);
  for (int i = 0; i < P->rows(); i++)
    for (SpMatCSR::InnerIterator it(*P, i); it; ++it)
      fprintf(fp, "%d %d %lf\n", i, it.col(), it.value());

  return;
}

/**
 * \brief Scans an Eigen CSC matrix from a Pajek file.
 *
 * Scans an Eigen CSC matrix from a Pajek file
 * (see <a href="http://mrvar.fdv.uni-lj.si/pajek/">Pajek documentation</a>).
 * \param[in] fp    Descriptor of the file to which to scan.
 * \return    Eigen matrix scanned.
 */
SpMatCSC *
pajek2Eigen(FILE *fp){
  char label[20];
  int N, E;
  int i, j, i0;
  double x;

  std::vector<Tri> tripletList;

  // Read vertices
  fscanf(fp, "%s %d", label, &N);

  // Define sparse matrix
  SpMatCSC *T = new SpMatCSC(N, N);

  // Read first (assume monotonous)
  fscanf(fp, "%d %s", &i0, label);
  for (int k = 1; k < N; k++){
    fscanf(fp, "%d %s", &i, label);
  }

  // Read Edges
  fscanf(fp, "%s %d", label, &E);

  // Reserve triplet capacity
  tripletList.reserve(E);

  for (int k = 0; k < E; k++){
    fscanf(fp, "%d %d %lf", &i, &j, &x);
    tripletList.push_back(Tri(i - i0, j - i0, x));
  }
  
  // Assign matrix elements
  T->setFromTriplets(tripletList.begin(), tripletList.end());

  return T;
}

/**
 * \brief Print an Eigen CSC matrix to file in compressed format.
 *
 * Print an Eigen CSC matrix in compressed format (see atio.hpp documentation).
 * \param[in] fp    Descriptor of the file to which to print.
 * \param[in] P    Eigen matrix to print.
 */
void
Eigen2Compressed(FILE *fp, SpMatCSC *P){
  char sparseType[] = "CSC";

  // Write type, inner size, outer size and number of non-zeros
  fprintf(fp, "%s %d %d %d\n", sparseType, P->innerSize(), P->outerSize(), P->nonZeros());

  // Write values
  for (int nz = 0; nz < P->nonZeros(); nz++)
    fprintf(fp, "%lf ", (P->valuePtr())[nz]);
  fprintf(fp, "\n");

  // Write row indices
  for (int nz = 0; nz < P->nonZeros(); nz++)
    fprintf(fp, "%d ", (P->innerIndexPtr())[nz]);
  fprintf(fp, "\n");

  // Write first element of column pointer
  for (int outer = 0; outer < P->outerSize()+1; outer++)
    fprintf(fp, "%d ", (P->outerIndexPtr())[outer]);
  fprintf(fp, "\n");
  
  return;
}

/**
 * \brief Print an Eigen CSR matrix to file in compressed format.
 *
 * Print an Eigen CSR matrix in compressed format (see atio.hpp documentation).
 * \param[in] fp    Descriptor of the file to which to print.
 * \param[in] P    Eigen matrix to print.
 */
void
Eigen2Compressed(FILE *fp, SpMatCSR *P){
  char sparseType[] = "CSR";

  // Write type, inner size, outer size and number of non-zeros
  fprintf(fp, "%s %d %d %d\n", sparseType, P->innerSize(), P->outerSize(), P->nonZeros());

  // Write values
  for (int nz = 0; nz < P->nonZeros(); nz++)
    fprintf(fp, "%lf ", (P->valuePtr())[nz]);
  fprintf(fp, "\n");

  // Write column indices
  for (int nz = 0; nz < P->nonZeros(); nz++)
    fprintf(fp, "%d ", (P->innerIndexPtr())[nz]);
  fprintf(fp, "\n");

  // Write first element of row pointer
  for (int outer = 0; outer < P->outerSize()+1; outer++)
    fprintf(fp, "%d ", (P->outerIndexPtr())[outer]);
  fprintf(fp, "\n");
  
  return;
}

/**
 * \brief Scan an Eigen CSR matrix from a file in compressed format.
 *
 * Scan an Eigen CSR matrix from a matrix file in compressed format (see atio.hpp documentation).
 * \param[in] fp    Descriptor of the file to which to scan.
 * \return Scanned Eigen matrix.
 */
SpMatCSR *
Compressed2Eigen(FILE *fp)
{
  int innerSize, outerSize, nnz;
  char sparseType[4];
  double *nzval;
  int *innerIndexPtr, *outerIndexPtr;
  SpMatCSR *T;
  std::vector<Tri> tripletList;

  // Read type, inner size, outer size and number of non-zeros and allocate
  fscanf(fp, "%s %d %d %d", sparseType, &outerSize, &innerSize, &nnz);
  nzval = new double [nnz];
  innerIndexPtr = new int [nnz];
  outerIndexPtr = new int[outerSize+1];
  T = new SpMatCSR(outerSize, innerSize);
  Eigen::VectorXf innerNNZ(outerSize);

  // Read values
  for (int nz = 0; nz < nnz; nz++)
    fscanf(fp, "%lf ", &nzval[nz]);

  // Read inner indices (column)
  for (int nz = 0; nz < nnz; nz++)
    fscanf(fp, "%d ", &innerIndexPtr[nz]);

  // Read first element of column pointer
  fscanf(fp, "%d ", &outerIndexPtr[0]);
  for (int outer = 1; outer < outerSize+1; outer++){
    fscanf(fp, "%d ", &outerIndexPtr[outer]);
    innerNNZ(outer-1) = outerIndexPtr[outer] - outerIndexPtr[outer-1];
  }
  T->reserve(innerNNZ);

  // Insert elements
  for (int outer = 0; outer < outerSize; outer++)
     for (int nzInner = outerIndexPtr[outer]; nzInner < outerIndexPtr[outer+1]; nzInner++)
       T->insertBackUncompressed(outer, innerIndexPtr[nzInner]) =  nzval[nzInner];
  //T.sumupDuplicates();

  delete nzval;
  delete innerIndexPtr;
  delete outerIndexPtr;
  
  return T;
}

/**
 * \brief Scan an edge list from a file in compressed format.
 *
 * Scan an edge list as a GSL matrix from a matrix file in compressed format
 * (see atio.hpp documentation).
 * \param[in] fp    Descriptor of the file to which to scan.
 * \return    Edge list as a GSL matrix
 */
gsl_matrix *
Compressed2EdgeList(FILE *fp)
{
  int innerSize, outerSize, nnz;
  char sparseType[4];
  double *nzval;
  int *idx, *ptr;
  gsl_matrix *EdgeList;
  int iOuter, iInner;
  
  // Read type, inner size, outer size and number of non-zeros and allocate
  fscanf(fp, "%s %d %d %d", sparseType, &innerSize, &outerSize, &nnz);
  nzval = new double [nnz];
  idx = new int [nnz];
  ptr = new int[outerSize+1];
  EdgeList = gsl_matrix_alloc(nnz, 3);
  if (strcmp(sparseType, "CSR") == 0)
    iOuter = 0;
  else if (strcmp(sparseType, "CSR") == 0)
    iOuter = 1;
  else{
    std::cerr << "Invalid sparse matrix type." << std::endl;
    exit(EXIT_FAILURE);
  }
  iInner = (iOuter + 1)%2;

  // Read values
  for (int nz = 0; nz < nnz; nz++)
    fscanf(fp, "%lf ", &nzval[nz]);

  // Read row indices
  for (int nz = 0; nz < nnz; nz++)
    fscanf(fp, "%d ", &idx[nz]);

  // Read first element of column pointer
  for (int outer = 0; outer < outerSize+1; outer++)
    fscanf(fp, "%d ", &ptr[outer]);

  int nz = 0;
  for (int outer = 0; outer < outerSize; outer++){
    for (int inner = ptr[outer]; inner < ptr[outer+1]; inner++){
	gsl_matrix_set(EdgeList, nz, iOuter, outer);
	gsl_matrix_set(EdgeList, nz, iInner, idx[inner]);
	gsl_matrix_set(EdgeList, nz, 2, nzval[inner]);
	nz++;
    }
  }

  delete(nzval);
  delete(idx);
  delete(ptr);
  return EdgeList;
}

/**
 * \brief Print and edge list to a file in compressed format.
 *
 * Print an edge list to a matrix file in compressed format (see atio.hpp documentation).
 * \param[in] src    Descriptor of the file to which to scan in compressed matrix format.
 * \param[in] dst    Descriptor of the file to which to print in edge list format.
 */
void
Compressed2EdgeList(FILE *src, FILE *dst)
{
  gsl_matrix *EdgeList = Compressed2EdgeList(src);
  size_t nnz = EdgeList->size1;

  for (size_t nz = 0; nz < nnz; nz++)
    fprintf(dst, "%d\t%d\t%lf\n",
	    (int) gsl_matrix_get(EdgeList, nz, 0),
	    (int) gsl_matrix_get(EdgeList, nz, 1),
	    gsl_matrix_get(EdgeList, nz, 2));

  gsl_matrix_free(EdgeList);
  return;
}

/**
 * \brief Convert an Eigen CSC matrix to an Eigen CSR matrix.
 *
 * Convert an Eigen CSC matrix to an Eigen CSR matrix.
 * \param[in] T    Eigen matrix from which to convert.
 * \return Eigen matrix converted.
 */
SpMatCSR *
CSC2CSR(SpMatCSC *T){
  int N = T->rows();

  // Define sparse matrix
  SpMatCSR *TCSR = new SpMatCSR(N, N);

  // Get triplet list
  std::vector<Tri> tripletList = Eigen2Triplet(T);

  // Assign matrix elements
  TCSR->setFromTriplets(tripletList.begin(), tripletList.end());

  return TCSR;
}

/**
 * \brief Convert an Eigen CSR matrix to an Eigen CSC matrix.
 *
 * Converts an Eigen CSR matrix to an Eigen CSC matrix.
 * \param[in] T    Eigen matrix from which to convert.
 * \return Eigen matrix converted.
 */
SpMatCSC *
CSR2CSC(SpMatCSR *T){
  int N = T->rows();

  // Define sparse matrix
  SpMatCSC *TCSC = new SpMatCSC(N, N);

  // Get triplet list
  std::vector<Tri> tripletList = Eigen2Triplet(T);

  // Assign matrix elements
  TCSC->setFromTriplets(tripletList.begin(), tripletList.end());

  return TCSC;
}

/**
 * \brief Convert an Eigen CSC matrix to a vector of Eigen triplet.
 *
 * Convert an Eigen CSC matrix to a vector of Eigen triplet.
 * \param[in] T    Eigen matrix from which to convert.
 * \return vector of Eigen triplet converted.
 */
std::vector<Tri>
Eigen2Triplet(SpMatCSC *T)
{
  // Works for column major only
  int nnz = T->nonZeros();
  std::vector<Tri> tripletList;
  tripletList.reserve(nnz);

  for (int beta = 0; beta < T->outerSize(); ++beta)
    for (SpMatCSC::InnerIterator it(*T, beta); it; ++it)
      tripletList.push_back(Tri(it.row(), it.col(), it.value()));

  return tripletList;
}

/**
 * \brief Convert an Eigen CSR matrix to a vector of Eigen triplet.
 *
 * Convert an Eigen CSR matrix to a vector of Eigen triplet.
 * \param[in] T    Eigen matrix from which to convert.
 * \return vector of Eigen triplet converted.
 */
std::vector<Tri>
Eigen2Triplet(SpMatCSR *T)
{
  // Works for column major only
  int nnz = T->nonZeros();
  std::vector<Tri> tripletList;
  tripletList.reserve(nnz);

  for (int beta = 0; beta < T->outerSize(); ++beta)
    for (SpMatCSR::InnerIterator it(*T, beta); it; ++it)
      tripletList.push_back(Tri(it.row(), it.col(), it.value()));

  return tripletList;
}

/**
 * \brief Print an Eigen CSR matrix as a dense matrix.
 *
 * Print an Eigen CSR matrix as a dense matrix with zero elements marked as "---".
 * \param[in] fp File to which to print.
 * \param[in] T Eigen matrix to print.
 * \param[in] format Format in which to print each matrix element.
 */
void
fprintfEigen(FILE *fp, SpMatCSR *T, const char *format)
{
  int count;
  for (int irow = 0; irow < T->outerSize(); ++irow){
    count = 0;
    for (SpMatCSR::InnerIterator it(*T, irow); it; ++it){
      while (count < it.col()){
	fprintf(fp, "---\t");
	count++;
      }
      fprintf(fp, format, it.value());
      fprintf(fp, "\t");
      count++;
    }
    while (count < T->innerSize()){
      fprintf(fp, "---\t");
      count++;
    }
    fprintf(fp, "\n");
  }
  return;
}

/**
 * \brief Count the number of lines in a file.
 *
 * Count the number of lines in a file.
 * \param[in] fp File from which to count lines.
 * \return Number of lines in file.
 */
size_t
lineCount(FILE *fp)
{
  size_t count = 0;
  int ch;

  // Count lines
  do {
    ch = fgetc(fp);
    if (ch == '\n')
      count++;
  } while (ch != EOF);

  return count;
}

#endif
