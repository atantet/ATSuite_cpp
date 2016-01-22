#Ifndef SDESOLVERS_HPP
#define SDESOLVERS_HPP

#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>
#include <ODESolvers.hpp>


/** \file SDESolvers.hpp
 *  \brief Solve stocahstic differential equations.
 *   
 *  Library to solve stochastic differential equations.
 *  The library uses polymorphism to design a
 *  stochastic model (class stochasticModel) from building blocks.
 *  Those building blocks are the vector field (class vectorField),
 *  inherited from class model
 *  and a stochastic numerical scheme (class stochasticNumericalScheme),
 *  and a stochastic vector field (class stochasticVectorField).
 */


/*
 * Class declarations:
 */

/** \brief Abstract stochastic vector field class.
 * 
 *  Abstract stochastic vector field class inheriting
 *  from the ordinary vector field class.
 */
class stochasticVectorField : public vectorField {
  gsl_rng *rng;              //!< Random number gnerator
  gsl_vector *noiseState;    //!< Current noise state (mainly a workspace)
public:
  /** \brief Default constructor. */
  stochasticVectorField() {}
  
  /** \brief Constructor setting the dimension, the generator and allocating. */
  stochasticVectorField(const size_t dim_, gsl_rng *rng_)
    : vectorField(dim_), rng(rng_)
  { noiseState = gsl_vector_alloc(dim); }
  
  /** \brief Destructor freeing noise. */
  ~stochasticVectorfield() { gsl_vector_free(noiseState); }
  
  /** Update noise realization. */
  void stepForwardNoise() {
    for (size_t i = 0; i < dim; i++)
      gsl_vector_set(noiseState, i, gsl_ran_gaussian(r, 1.));
  }

  /** \brief Virtual method for evaluating the vector field at a given state. */
  virtual void evalField(gsl_vector *state, gsl_vector *field) = 0;
};


/** \brief Additive Wiener process.
 *
 *  Additive Wiener process stochastic vector field.
 */
class additiveWiener : public stochasticVectorField {
  gsl_matrix *Q;  //!< Correlation matrix to apply to noise realization.
public:
  /** \brief Default constructor. */
  additiveWiener(){}
  
  /** \brief Construction by copying the matrix of the linear operator. */
  additiveWiener(const gsl_matrix *Q_, gsl_rng *rng_)
    : stochasticVectorField(A_->size1, rng_)
  { gsl_matrix_memcpy(Q, Q_); }
  
  /** Destructor freeing the matrix. */
  ~additiveWiener(){ gsl_matrix_free(Q); }

  /** \brief Return the parameters of the model. */
  void getParameters(gsl_matrix *Q_) { gsl_matrix_memcpy(Q_, Q); return; }

  /** \brief Set parameters of the model. */
  void setParameters(const gsl_matrix *Q_) { gsl_matrix_memcpy(Q, Q_); return; }

  /** \brief Evaluate the linear vector field at a given state. */
  void evalField(gsl_vector *state, gsl_vector *field);
};


/** \brief Linear multiplicative Wiener process.
 * 
 *  Linear multiplicative Wiener process stochastic vector field.
 *  Note: nondiagonal state multiplication not (yet) implemented.
 */
class multiplicativeLinearWiener : public stochasticVectorField {
  gsl_matrix *Q;   //!< Correlation matrix to apply to noise realization.
public:
  /** \brief Default constructor. */
  multiplicativeLinearWiener(){}
  
  /** \brief Construction by copying the matrix of the linear operator. */
  multiplicativeLinearWiener(const gsl_matrix *Q_, gsl_rng *rng_)
    : stochasticVectorField(A_->size1, rng_)
  { gsl_matrix_memcpy(Q, Q_); }
  
  /** Destructor freeing the matrix. */
  ~multiplicativeLinearWiener(){ gsl_matrix_free(Q); }

  /** \brief Return the parameters of the model. */
  void getParameters(gsl_matrix *Q_) { gsl_matrix_memcpy(Q_, Q); return; }

  /** \brief Set parameters of the model. */
  void setParameters(const gsl_matrix *Q_) { gsl_matrix_memcpy(Q, Q_); return; }

  /** \brief Evaluate the linear vector field at a given state. */
  void evalField(gsl_vector *state, gsl_vector *field);
};


/** \brief Abstract stochastic numerical scheme class.
 *
 *  Abstract stochastic numerical scheme class inheriting
 *  from the ordinary numerical scheme class.
 */
class stochasticNumericalScheme : public numericalScheme {
public:
    /** \brief Default constructor. */
  stochasticNumericalScheme() {}
  
  /** \brief Constructor defining the dimensions, time step and allocating workspace. */
  stochasticNumericalScheme(const size_t dim_, const size_t dimWork_, const double dt_)
    : numericalScheme(dim_, dimWork_, dt_) {}
  
  /** \brief Destructor freeing workspace. */
  ~stochasticNumericalScheme() {}

  /** \brief Virtual method to integrate the stochastic model one step forward. */
  virtual void stepForward(const vectorField *field,
			   const stochasticVectorField *stocField,
			   gsl_vector *currentState) = 0;
};


/** \brief Euler-Maruyama stochastic numerical scheme.
 *  Euler-Maruyama stochastic numerical scheme.
 */
class EulerMaruyama : public stochasticNumericalScheme {
public:
    /** \brief Default constructor. */
  EulerMaruyama() {}
  
  /** \brief Constructor defining the dimensions, time step and allocating workspace. */
  EulerMaruyama(const size_t dim_, const double dt_)
    : stochasticNumericalScheme(dim_, 2, dt_) {}
  
  /** \brief Destructor freeing workspace. */
  ~EulerMaruyama() {}

  /** \brief Virtual method to integrate the stochastic model one step forward. */
  virtual void stepForward(const vectorField *field,
			   const stochasticVectorField *stocField,
			   gsl_vector *currentState);
};


/** \brief Stochastic model class.
 * 
 * Stochstic model class.
 *  A stochastic model is defined by a vector field,
 *  a stochastic vector field and a numerical scheme.
 *  The current state of the model is also recorded.
 *  Attention: the constructors do not copy the vector field
 *  and the numerical scheme given to them, so that
 *  any modification or freeing will affect the model.
 */

class stochasticModel : public model {
  const stochasticVectorField *stocField; //!< Stochastic vector field (diffusion)
  
public:
  /** \brief Default constructor */
  stochasticModel() {}

  /** \brief Constructor assigning a vector field, a numerical scheme
   *  and a stochastic vector field and setting initial state to origin. */
  stochasticModel(vectorField *field_, numericalScheme *scheme_,
		  stochasticVectorField *stocField_)
    : model(field_, scheme_), stocField(stocField_) {}

  /** \brief Constructor assigning a vector field, a numerical scheme
   *  and a stochastic vector field and setting initial state. */
  stochasticModel(vectorField *field_, numericalScheme *scheme_,
		  stochasticVectorField *stocField_, gsl_vector *initState)
    : model(field_, scheme_, initState_), stocField(stocField_) {}

  /** \brief Destructor freeing memory. */
  ~stochasticModel() {}

  /** \brief One time-step forward stochastic Integration of the model. */
  void stepForward();
};
  

/**
 * Method definitions
 */

void
additiveWiener::evalField(gsl_vector *state, gsl_vector *field)
{
  // Get new noise realization
  stepForwardNoise();
    
  // Wiener: apply correlation matrix Q to noise realization
  gsl_blas_dgemv(CblasNoTrans, 1., Q, noiseState, 0., field);

  // Multiply state
  gsl_vector_mul(field, state);

  return;
}


void
multiplicativeLinearWiener::evalField(gsl_vector *state, gsl_vector *field)
{
  // Get new noise realization
  stepForwardNoise();
  
  // Additive Wiener: apply correlation matrix Q to noise realization
  gsl_blas_dgemv(CblasNoTrans, 1., Q, noiseState, 0., field);

  return;
}


void
EulerMaruyama::stepForward(const vectorField *field,
			   const stochasticVectorField *stocField,
			   gsl_vector *currentState)
{
  gsl_vector_view tmp = gsl_matrix_row(work, 0);
  gsl_vector_view tmp1 = gsl_matrix_row(work, 1);

  // Evalueate fields
  field->evalField(currentState, &tmp.vector);
  stocField->evalField(currentState, &tmp1.vector);

  // Add drift
  gsl_vector_scale(&tmp.vector, dt);
  gsl_vector_add(currentState, &tmp.vector);

  // Add diffusion
  gsl_vector_scale(&tmp1.vector, sqrt(dt));
  gsl_vector_add(currentState, &tmp2.vector);

  return;
}


/**
 * Integrate one step forward the stochastic model
 * by calling the numerical scheme.
 */
void
stochasticModel::stepForward()
{
  // Apply stochastic numerical scheme to step forward
  scheme.stepForward(field, stocField, currentState);
    
  return;
}


#endif
