#ifndef ODESOLVERS_HPP
#define ODESOLVERS_HPP

#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>

/** \file ODESolvers.hpp
 *  \brief Ordinary differential equation solvers.
 *   
 *  ATSuite ordinary differential equation solvers.
 */

// Declarations
gsl_matrix * generateCuspEuler(gsl_vector *, double, double, double, double,
			       int, double);
gsl_vector * cuspEuler(gsl_vector *, double, double, double);
gsl_vector * cuspField(gsl_vector *, double, double);
gsl_matrix * generateLorenzRK4(gsl_vector *, double, double, double, double,
			       double, int, double);
gsl_vector * lorenzRK4(gsl_vector *, double, double, double, double);
gsl_vector * lorenzField(gsl_vector *, double, double, double);


// Definitions
/**
 * \brief Integrate the cusp normal form with an Euler scheme.
 *
 * Integrate the cusp normal form (see Strogatz, 1994) with an Euler scheme.
 * \param[in] state    Initial state.
 * \param[in] r        Parameter \f$r\f$.
 * \param[in] h        Parameter \f$h\f$.
 * \param[in] length   Length of integration.
 * \param[in] dt       Time step.
 * \param[in] sampling Step between each sampled state.
 * \param[in] spinup   Lenght of initial spinup period to discard.
 * \return             GSL matrix recording the integration.
 */
gsl_matrix * generateCuspEuler(gsl_vector *state, double r, double h,
			       double length, double dt, int sampling,
			       double spinup){
  size_t nt = length / dt;
  size_t ntSpinup = spinup / dt;
  size_t dim = state->size;
  gsl_matrix *data = gsl_matrix_alloc((size_t) (nt/sampling), dim);
  gsl_vector *res, *init;

  init = gsl_vector_alloc(dim);
  gsl_vector_memcpy(init, state);
  // Get spinup
  for (size_t i = 1; i <= ntSpinup; i++){
    res = cuspEuler(init, r, h, dt);
    gsl_vector_memcpy(init, res);
    gsl_vector_free(res);
  }
  // Get record
  for (size_t i = ntSpinup+1; i <= nt; i++){
    res = cuspEuler(init, r, h, dt);
    if (i%sampling == 0)
      gsl_matrix_set_row(data, (i-ntSpinup)/sampling-1, res);
    gsl_vector_memcpy(init, res);
    gsl_vector_free(res);
  }
  gsl_vector_free(init);
  
  return data;
}

/**
 * \brief Integrate one step forward the cusp normal form with an Euler scheme.
 *
 * Integrate one step forward the cusp normal form (see Strogatz, 1994) with an Euler scheme.
 * \param[in] state    Present state.
 * \param[in] r        Parameter r.
 * \param[in] h        Parameter h.
 * \param[in] dt       Time step.
 * \return             GSL vector of the future state.
 */
gsl_vector * cuspEuler(gsl_vector *state,
		       double r, double h,
		       double dt)
{
  size_t dim = state->size;
  gsl_vector *newState = gsl_vector_calloc(dim);

  newState = cuspField(state, r, h);
  gsl_vector_scale(newState, dt);
  gsl_vector_add(newState, state);

  return newState;
}

/**
 * \brief Get the field of the normal form of the cusp at a given state.
 *
 * Get the field of the normal form of the cusp at a given state.
 * \param[in] state    State.
 * \param[in] r        Parameter r.
 * \param[in] h        Parameter h.
 * \return             GSL vector of the field at the state.
 */
gsl_vector * cuspField(gsl_vector *state,
		       double r, double h)
{
  size_t dim = state->size;
  gsl_vector *field = gsl_vector_alloc(dim);

  // Fx = sigma * (y - x)
  gsl_vector_set(field, 0, h + r * gsl_vector_get(state, 0) 
		 - pow(gsl_vector_get(state, 0), 3));
  
  return field;
}

/**
 * \brief Integrate the Lorenz, 1963 with an RK 4 scheme.
 *
 * Integrate the Lorenz, 1963 with an Runge-Kutta 4 scheme.
 * \param[in] state    Initial state.
 * \param[in] rho      Parameter \f$\rho\f$.
 * \param[in] sigma    Parameter \f$\sigma\f$.
 * \param[in] beta     Parameter \f$\beta\f$.
 * \param[in] length   Length of integration.
 * \param[in] dt       Time step.
 * \param[in] sampling Step between each sampled state.
 * \param[in] spinup   Lenght of initial spinup period to discard.
 * \return             GSL matrix recording the integration.
 */
gsl_matrix * generateLorenzRK4(gsl_vector *state,
			       double rho, double sigma, double beta,
			       double length, double dt, int sampling,
			       double spinup)
{
  size_t nt = length / dt;
  size_t ntSpinup = spinup / dt;
  size_t dim = state->size;
  gsl_matrix *data = gsl_matrix_alloc((size_t) (nt/sampling), dim);
  gsl_vector *res, *init;

  init = gsl_vector_alloc(dim);
  gsl_vector_memcpy(init, state);
  // Get spinup
  for (size_t i = 1; i <= ntSpinup; i++){
    res = lorenzRK4(init, rho, sigma, beta, dt);
    gsl_vector_memcpy(init, res);
    gsl_vector_free(res);
  }
  // Get record
  for (size_t i = ntSpinup+1; i <= nt; i++){
    res = lorenzRK4(init, rho, sigma, beta, dt);
    if (i%sampling == 0)
      gsl_matrix_set_row(data, (i-ntSpinup)/sampling-1, res);
    gsl_vector_memcpy(init, res);
    gsl_vector_free(res);
  }
  gsl_vector_free(init);
  
  return data;
}

/**
 * \brief Integrate one step forward the Lorenz, 1963 with an RK4 scheme.
 *
 * Integrate one step forward the Lorenz, 1963 with an Runge-Kutta 4 scheme.
 * \param[in] state    Present state.
 * \param[in] rho      Parameter \f$\rho\f$.
 * \param[in] sigma    Parameter \f$\sigma\f$.
 * \param[in] beta     Parameter \f$\beta\f$.
 * \param[in] dt       Time step.
 * \return             GSL vector of the future state.
 */
gsl_vector * lorenzRK4(gsl_vector *state,
		       double rho, double sigma, double beta,
		       double dt)
{
  size_t dim = state->size;
  gsl_vector *k1, *k2, *k3, *k4;
  gsl_vector *tmp = gsl_vector_calloc(dim);

  k1 = lorenzField(state, rho, sigma, beta);
  gsl_vector_scale(k1, dt);
  
  gsl_vector_memcpy(tmp, k1);
  gsl_vector_scale(tmp, 0.5);
  gsl_vector_add(tmp, state);
  k2 = lorenzField(tmp, rho, sigma, beta);
  gsl_vector_scale(k2, dt);

  gsl_vector_memcpy(tmp, k2);
  gsl_vector_scale(tmp, 0.5);
  gsl_vector_add(tmp, state);
  k3 = lorenzField(tmp, rho, sigma, beta);
  gsl_vector_scale(k3, dt);

  gsl_vector_memcpy(tmp, k3);
  gsl_vector_add(tmp, state);
  k4 = lorenzField(tmp, rho, sigma, beta);
  gsl_vector_scale(k4, dt);

  gsl_vector_scale(k2, 2);
  gsl_vector_scale(k3, 2);
  gsl_vector_memcpy(tmp, k1);
  gsl_vector_add(tmp, k2);
  gsl_vector_add(tmp, k3);
  gsl_vector_add(tmp, k4);
  gsl_vector_scale(tmp, 1. / 6);
  gsl_vector_add(tmp, state);

  gsl_vector_free(k1);
  gsl_vector_free(k2);
  gsl_vector_free(k3);
  gsl_vector_free(k4);

  return tmp;
}

/**
 * \brief Get the field of the Lorenz, 1963 at a given state.
 *
 * Get the field of the Lorenz, 1963 at a given state.
 * \param[in] state    State.
 * \param[in] rho      Parameter \f$\rho\f$.
 * \param[in] sigma    Parameter \f$\sigma\f$.
 * \param[in] beta     Parameter \f$\beta\f$.
 * \return             GSL vector of the field at the state.
 */
gsl_vector * lorenzField(gsl_vector *state,
			 double rho, double sigma, double beta)
{
  size_t dim = state->size;
  gsl_vector *field = gsl_vector_alloc(dim);

  // Fx = sigma * (y - x)
  gsl_vector_set(field, 0, sigma * (gsl_vector_get(state, 1) - gsl_vector_get(state, 0)));
  // Fy = x * (rho - z) - y
  gsl_vector_set(field, 1, gsl_vector_get(state, 0)
		 * (rho - gsl_vector_get(state, 2)) - gsl_vector_get(state, 1));
  // Fz = x*y - beta*z
  gsl_vector_set(field, 2, gsl_vector_get(state, 0) * gsl_vector_get(state, 1)
		 - beta * gsl_vector_get(state, 2));
  
  return field;
}

#endif
