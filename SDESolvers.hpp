#Ifndef SDESOLVERS_HPP
#define SDESOLVERS_HPP

#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <ODESolvers.hpp>

/** \file SDESolvers.hpp
 *  \brief Stochastic differential equation solvers.
 *   
 *  ATSuite stochastic differential equation solvers.
 */

// Declarations
gsl_matrix * generateCuspAdditiveWienerEM(gsl_vector *, double, double,
					  gsl_matrix *, double,
					  double, double, int, double);
gsl_matrix * generateCuspAdditiveWienerEM(gsl_vector *, double,
					  gsl_vector *,
					  gsl_matrix *, double,
					  double, double, int, double);
gsl_vector * cuspAdditiveWienerEM(gsl_vector *, double, double,
				  gsl_vector *, double, double);
gsl_matrix * generateLorenzLinearWienerEM(gsl_vector *, double, double, double,
					  gsl_matrix *, double,
					  double, double, int, double);
gsl_vector * lorenzLinearWienerEM(gsl_vector *, double, double, double,
				  gsl_vector *, double, double);
gsl_vector * additiveWienerField(double, gsl_vector *);
gsl_vector * linearWienerField(gsl_vector *, double, gsl_vector *);


// Definitions

/**
 * \brief Integrate the cusp normal form with additive Wiener and EM scheme.
 *
 * Integrate the cusp normal form (see Strogatz, 1994)
 * with additive Wiener process, with an Euler-Maruyama scheme.
 * \param[in] state        Initial state.
 * \param[in] r            Parameter \f$r\f$.
 * \param[in] h            Parameter \f$h\f$.
 * \param[in] noiseSamples GSL matrix of noise realizations for each time step.
 * \param[in] Q            Noise intensity.
 * \param[in] length       Length of integration.
 * \param[in] dt           Time step.
 * \param[in] sampling     Step between each sampled state.
 * \param[in] spinup       Lenght of initial spinup period to discard.
 * \return                 GSL matrix recording the integration.
 */
gsl_matrix *
generateCuspAdditiveWienerEM(gsl_vector *state,
			     double r, double h,
			     gsl_matrix *noiseSamples, double Q,
			     double length, double dt,
			     int sampling, double spinup)
{
  size_t nt = length / dt;
  size_t ntSpinup = spinup / dt;
  size_t dim = state->size;
  gsl_matrix *data = gsl_matrix_alloc((size_t) (nt/sampling), dim);
  gsl_vector *newState, *initState, *noiseSample;

  initState = gsl_vector_alloc(dim);
  gsl_vector_memcpy(initState, state);
  
  // Get spinup
  for (size_t i = 1; i <= ntSpinup; i++){
    // Get noise sample
    noiseSample = gsl_vector_alloc(dim);
    gsl_matrix_get_row(noiseSample, noiseSamples, i);
    
    // Get new state
    newState = cuspAdditiveWienerEM(initState, r, h,
				    noiseSample, Q,
				    dt);

    gsl_vector_memcpy(initState, newState);
    gsl_vector_free(newState);
    gsl_vector_free(noiseSample);
  }
  
  // Get record
  for (size_t i = 1; i <= nt; i++){
    // Get noise sample
    noiseSample = gsl_vector_alloc(dim);
    gsl_matrix_get_row(noiseSample, noiseSamples, ntSpinup+i);
    
    // Get new state
    newState = cuspAdditiveWienerEM(initState, r, h,
				    noiseSample, Q, dt);

    // Save new state
    if (i%sampling == 0)
      gsl_matrix_set_row(data, i/sampling-1, newState);

    gsl_vector_memcpy(initState, newState);
    gsl_vector_free(newState);
    gsl_vector_free(noiseSample);
  }
  gsl_vector_free(initState);
  
  return data;
}

/**
 * \brief Transient int. of the cusp normal form with additive Wiener and EM scheme.
 *
 * Transient integration of the cusp normal form (see Strogatz, 1994)
 * with additive Wiener process, with an Euler-Maruyama scheme.
 * \param[in] state        Initial state.
 * \param[in] r            Parameter \f$r\f$.
 * \param[in] hTransient   GSL vector of the variable parameter \f$h(t)\f$ for each time step.
 * \param[in] noiseSamples GSL matrix of noise realizations for each time step.
 * \param[in] Q            Noise intensity.
 * \param[in] length       Length of integration.
 * \param[in] dt           Time step.
 * \param[in] sampling     Step between each sampled state.
 * \param[in] spinup       Lenght of initial spinup period to discard.
 * \return                 GSL matrix recording the integration.
 */
gsl_matrix *
generateCuspAdditiveWienerEM(gsl_vector *state,
			     double r, gsl_vector *hTransient,
			     gsl_matrix *noiseSamples, double Q,
			     double length, double dt,
			     int sampling, double spinup)
{
  size_t nt = length / dt;
  size_t ntSpinup = spinup / dt;
  size_t dim = state->size;
  gsl_matrix *data = gsl_matrix_alloc((size_t) (nt/sampling), dim);
  gsl_vector *newState, *initState, *noiseSample;
  double h;

  initState = gsl_vector_alloc(dim);
  gsl_vector_memcpy(initState, state);
  
  // Get spinup
  for (size_t i = 1; i <= ntSpinup; i++){
    // Get noise sample
    noiseSample = gsl_vector_alloc(dim);
    gsl_matrix_get_row(noiseSample, noiseSamples, i);

    // Get transient parameter value
    h = gsl_vector_get(hTransient, i);
    
    // Get new state
    newState = cuspAdditiveWienerEM(initState, r, h,
				    noiseSample, Q,
				    dt);

    gsl_vector_memcpy(initState, newState);
    gsl_vector_free(newState);
    gsl_vector_free(noiseSample);
  }
  
  // Get record
  for (size_t i = 1; i <= nt; i++){
    // Get noise sample
    noiseSample = gsl_vector_alloc(dim);
    gsl_matrix_get_row(noiseSample, noiseSamples, ntSpinup+i);
    
    // Get transient parameter value
    h = gsl_vector_get(hTransient, i);
    
    // Get new state
    newState = cuspAdditiveWienerEM(initState, r, h,
				    noiseSample, Q, dt);

    // Save new state
    if (i%sampling == 0)
      gsl_matrix_set_row(data, i/sampling-1, newState);

    gsl_vector_memcpy(initState, newState);
    gsl_vector_free(newState);
    gsl_vector_free(noiseSample);
  }
  gsl_vector_free(initState);
  
  return data;
}


/**
 * \brief Integrate one step the cusp normal form with an additive Wiener and EM scheme.
 *
 * Integrate one step forward the cusp normal form (see Strogatz, 1994)
 * with an additive Wiener process, with an Euler-Maruyama scheme.
 * \param[in] state        Present state.
 * \param[in] r            Parameter r.
 * \param[in] h            Parameter h.
 * \param[in] noiseSample  GSL matrix of noise realizations for each time step.
 * \param[in] Q            Noise intensity.
 * \param[in] dt           Time step.
 * \return                 GSL vector of the future state.
 */
gsl_vector *
cuspAdditiveWienerEM(gsl_vector *state,
		     double r, double h,
		     gsl_vector *noiseSample, double Q,
		     double dt)
{
  size_t dim = state->size;
  gsl_vector *field, *newState;

  newState = gsl_vector_alloc(dim);
  gsl_vector_memcpy(newState, state);
    
  field = cuspField(state, r, h);
  gsl_vector_scale(field, dt);
  gsl_vector_add(newState, field);
  gsl_vector_free(field);
  
  field = additiveWienerField(Q, noiseSample);
  gsl_vector_scale(field, sqrt(dt));
  gsl_vector_add(newState, field);
  gsl_vector_free(field);

  return newState;
}

/**
 * \brief Integrate the Lorenz, 1963 with multiplicative linear Wiener and EM scheme.
 *
 * Integrate the Lorenz, 1963 with multiplicative linear Wiener process,
 * with an Euler-Maruyama Scheme.
 * \param[in] state        Initial state.
 * \param[in] rho          Parameter \f$\rho\f$.
 * \param[in] sigma        Parameter \f$\sigma\f$.
 * \param[in] beta         Parameter \f$\beta\f$.
 * \param[in] noiseSamples GSL matrix of noise realizations for each time step.
 * \param[in] Q            Noise intensity.
 * \param[in] length       Length of integration.
 * \param[in] dt           Time step.
 * \param[in] sampling     Step between each sampled state.
 * \param[in] spinup       Lenght of initial spinup period to discard.
 * \return                 GSL matrix recording the integration.
 */
gsl_matrix *
generateLorenzLinearWienerEM(gsl_vector *state, double rho,
			     double sigma, double beta,
			     gsl_matrix *noiseSamples, double Q,
			     double length, double dt,
			     int sampling, double spinup)
{
  size_t nt = length / dt;
  size_t ntSpinup = spinup / dt;
  size_t dim = state->size;
  gsl_matrix *data = gsl_matrix_alloc((size_t) (nt/sampling), dim);
  gsl_vector *newState, *initState, *noiseSample;

  initState = gsl_vector_alloc(dim);
  gsl_vector_memcpy(initState, state);
  
  // Get spinup
  for (size_t i = 1; i <= ntSpinup; i++){
    // Get noise sample
    noiseSample = gsl_vector_alloc(dim);
    gsl_matrix_get_row(noiseSample, noiseSamples, i);
    
    // Get new state
    newState = lorenzLinearWienerEM(initState, rho, sigma, beta,
			       noiseSample, Q,
			       dt);

    gsl_vector_memcpy(initState, newState);
    gsl_vector_free(newState);
    gsl_vector_free(noiseSample);
  }
  
  // Get record
  for (size_t i = 1; i <= nt; i++){
    // Get noise sample
    noiseSample = gsl_vector_alloc(dim);
    gsl_matrix_get_row(noiseSample, noiseSamples, ntSpinup+i);
    
    // Get new state
    newState = lorenzLinearWienerEM(initState, rho, sigma, beta,
			       noiseSample, Q, dt);

    // Save new state
    if (i%sampling == 0)
      gsl_matrix_set_row(data, i/sampling-1, newState);

    gsl_vector_memcpy(initState, newState);
    gsl_vector_free(newState);
    gsl_vector_free(noiseSample);
  }
  gsl_vector_free(initState);
  
  return data;
}

/**
 * \brief Integrate one step the Lorenz, 1963 with multiplicative linear Wiener,
 * and EM Scheme.
 *
 * Integrate one step the Lorenz, 1963 with multiplicative linear Wiener process,
 * with an Euler-Maruyama Scheme.
 * \param[in] state        Initial state.
 * \param[in] rho          Parameter \f$\rho\f$.
 * \param[in] sigma        Parameter \f$\sigma\f$.
 * \param[in] beta         Parameter \f$\beta\f$.
 * \param[in] noiseSample  GSL matrix of noise realizations for each time step.
 * \param[in] Q            Noise intensity.
 * \param[in] dt           Time step.
 * \return                 GSL matrix recording the integration.
 */
gsl_vector * lorenzLinearWienerEM(gsl_vector *state,
				  double rho, double sigma, double beta,
				  gsl_vector *noiseSample, double Q,
				  double dt)
{
  size_t dim = state->size;
  gsl_vector *field, *newState;

  newState = gsl_vector_alloc(dim);
  gsl_vector_memcpy(newState, state);
    
  field = lorenzField(state, rho, sigma, beta);
  gsl_vector_scale(field, dt);
  gsl_vector_add(newState, field);
  gsl_vector_free(field);
  
  field = linearWienerField(state, Q, noiseSample);
  gsl_vector_scale(field, sqrt(dt));
  gsl_vector_add(newState, field);
  gsl_vector_free(field);

  return newState;
}

/**
 * \brief Get the field of the additive Wiener.
 *
 * Get the field of the additive Wiener process.
 * \param[in] Q            Noise intensity.
 * \param[in] noiseSample  GSL vector of noise realization for this time step.
 * \return                 GSL vector of the field at the state.
 */
gsl_vector * additiveWienerField(double Q, gsl_vector *noiseSample)
{
  size_t dim = noiseSample->size;
  gsl_vector *field = gsl_vector_alloc(dim);

  gsl_vector_memcpy(field, noiseSample);
  gsl_vector_scale(field, Q);
  
  return field;
}

/**
 * \brief Get the field of the multiplicative linear Wiener.

 * Get the field of the multiplicative linear Wiener process.
 * \param[in] state        Present state.
 * \param[in] Q            Noise intensity.
 * \param[in] noiseSample  GSL vector of noise realization for this time step.
 * \return                 GSL vector of the field at the state.
 */
gsl_vector * linearWienerField(gsl_vector *state, double Q, gsl_vector *noiseSample)
{
  size_t dim = noiseSample->size;
  gsl_vector *field = gsl_vector_alloc(dim);

  gsl_vector_memcpy(field, noiseSample);
  gsl_vector_scale(field, Q);
  gsl_vector_mul(field, state);
  
  return field;
}

#endif
