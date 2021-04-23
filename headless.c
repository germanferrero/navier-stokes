/*
  ======================================================================
   demo.c --- protoype to show off the simple solver
  ----------------------------------------------------------------------
   Author : Jos Stam (jstam@aw.sgi.com)
   Creation Date : Jan 9 2003

   Description:

	This code is a simple prototype that demonstrates how to use the
	code provided in my GDC2003 paper entitles "Real-Time Fluid Dynamics
	for Games". This code uses OpenGL and GLUT for graphics and interface

  =======================================================================
*/

#include <stdio.h>
#include <stdlib.h>

#include "wtime.h"

/* macros */

#define IX(i, j) ((i) + (N + 2) * (j))

/* external definitions (from solver.c) */

extern void dens_step(int N, float* x, float* x0, float* u, float* v, float diff, float dt);
extern void vel_step(int N, float* u, float* v, float* u0, float* v0, float visc, float dt);

/* global variables */

static int N;
static float dt, diff, visc;
static float force, source;

static float *u, *v, *u_prev, *v_prev;
static float *dens, *dens_prev;


/*
  ----------------------------------------------------------------------
   free/clear/allocate simulation data
  ----------------------------------------------------------------------
*/


static void free_data(void)
{
    if (u) {
        free(u);
    }
    if (v) {
        free(v);
    }
    if (u_prev) {
        free(u_prev);
    }
    if (v_prev) {
        free(v_prev);
    }
    if (dens) {
        free(dens);
    }
    if (dens_prev) {
        free(dens_prev);
    }
}

static void clear_data(void)
{
    int i, size = (N + 2) * (N + 2);

    for (i = 0; i < size; i++) {
        u[i] = v[i] = u_prev[i] = v_prev[i] = dens[i] = dens_prev[i] = 0.0f;
    }
}

static int allocate_data(void)
{
    int size = (N + 2) * (N + 2);

    u = (float*)malloc(size * sizeof(float));
    v = (float*)malloc(size * sizeof(float));
    u_prev = (float*)malloc(size * sizeof(float));
    v_prev = (float*)malloc(size * sizeof(float));
    dens = (float*)malloc(size * sizeof(float));
    dens_prev = (float*)malloc(size * sizeof(float));

    if (!u || !v || !u_prev || !v_prev || !dens || !dens_prev) {
        fprintf(stderr, "cannot allocate data\n");
        return (0);
    }

    return (1);
}


static void react(float* d, float* u, float* v)
{
    int i, size = (N + 2) * (N + 2);
    float max_velocity2 = 0.0f;
    float max_density = 0.0f;

    max_velocity2 = max_density = 0.0f;
    for (i = 0; i < size; i++) {
        if (max_velocity2 < u[i] * u[i] + v[i] * v[i]) {
            max_velocity2 = u[i] * u[i] + v[i] * v[i];
        }
        if (max_density < d[i]) {
            max_density = d[i];
        }
    }

    for (i = 0; i < size; i++) {
        u[i] = v[i] = d[i] = 0.0f;
    }

    if (max_velocity2 < 0.0000005f) {
        u[IX(N / 2, N / 2)] = force * 10.0f;
        v[IX(N / 2, N / 2)] = force * 10.0f;
    }
    if (max_density < 1.0f) {
        d[IX(N / 2, N / 2)] = source * 10.0f;
    }

    return;
}

static void one_step(void)
{
    static int times = 1;
    static double start_t = 0.0;
    static double start_total_t = 0.0;
    static double total_cells_p_s = 0.0;
    static double one_second = 0.0;
    static double react_cells_p_s = 0.0;
    static double vel_cells_p_s = 0.0;
    static double dens_cells_p_s = 0.0;

    start_total_t = wtime();
    start_t = wtime();
    react(dens_prev, u_prev, v_prev);
    react_cells_p_s += (N * N) / (wtime() - start_t);

    start_t = wtime();
    vel_step(N, u, v, u_prev, v_prev, visc, dt);
    vel_cells_p_s += (N * N) / (wtime() - start_t);

    start_t = wtime();
    dens_step(N, dens, dens_prev, u, v, diff, dt);
    dens_cells_p_s += (N * N) / (wtime() - start_t);

    total_cells_p_s += (N * N) / (wtime() - start_total_t);
    if (1.0 < wtime() - one_second) { /* at least 1s between stats */
        fprintf(stderr, "%lf, %lf, %lf, %lf: cells per second total step, react, vel_step, dens_step\n",
                total_cells_p_s / times,
                react_cells_p_s / times, vel_cells_p_s / times, dens_cells_p_s / times);
        one_second = wtime();
        react_cells_p_s = 0.0;
        vel_cells_p_s = 0.0;
        dens_cells_p_s = 0.0;
        total_cells_p_s = 0.0;
        times = 1;
    } else {
        times++;
    }
}


/*
  ----------------------------------------------------------------------
   main --- main routine
  ----------------------------------------------------------------------
*/

int main(int argc, char** argv)
{
    int i = 0;
    static double start_t = 0.0;
    static double end_t = 0.0;

    if (argc != 1 && argc != 7) {
        fprintf(stderr, "usage : %s N dt diff visc force source\n", argv[0]);
        fprintf(stderr, "where:\n");
        fprintf(stderr, "\t N      : grid resolution\n");
        fprintf(stderr, "\t dt     : time step\n");
        fprintf(stderr, "\t diff   : diffusion rate of the density\n");
        fprintf(stderr, "\t visc   : viscosity of the fluid\n");
        fprintf(stderr, "\t force  : scales the mouse movement that generate a force\n");
        fprintf(stderr, "\t source : amount of density that will be deposited\n");
        exit(1);
    }

    if (argc == 1) {
        N = 128;
        dt = 0.1f;
        diff = 0.0f;
        visc = 0.0f;
        force = 5.0f;
        source = 100.0f;
        fprintf(stderr, "Using defaults : N=%d dt=%g diff=%g visc=%g force = %g source=%g\n",
                N, dt, diff, visc, force, source);
    } else {
        N = atoi(argv[1]);
        dt = atof(argv[2]);
        diff = atof(argv[3]);
        visc = atof(argv[4]);
        force = atof(argv[5]);
        source = atof(argv[6]);
    }

    if (!allocate_data()) {
        exit(1);
    }
    clear_data();

    for (i = 0; i < 8; i++) {
        if (i == 2) {
            start_t = wtime();
        }
        one_step();
    }
    end_t = wtime();
    printf("%lf\n", ((N * N) * (8 - 2)) / (end_t - start_t));
    free_data();

    exit(0);
}
