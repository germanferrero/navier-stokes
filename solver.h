//
// solver.h
//

#ifndef SOLVER_H_INCLUDED
#define SOLVER_H_INCLUDED

void dens_step(unsigned int n, float* x, float* x0, float* u, float* v, float diff, float dt);
void vel_step(unsigned int n, float* u, float* v, float* u0, float* v0, float visc, float dt);
void launcher_get_velocity2(float * velocity2, unsigned int n, const float* u, const float* v);
void launcher_add_forces(float *max_velocity2, unsigned int n, float force, float * u, float * v);
void launcher_add_densities(float *max_density, unsigned int n,  float source, float * d);

#endif /* SOLVER_H_INCLUDED */
