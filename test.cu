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
#include <cuda_runtime.h>
#include <cub/cub.cuh>

#include "cuda_helper.h"
#include "wtime.h"
#include "indices.h"
#include "solver.h"

/* macros */

#define IX(x,y) (rb_idx((x),(y),(N+2)))

/* global variables */

static int N;
static float dt, diff, visc;
static float force, source;

static float *u, *v, *u_prev, *v_prev;
static float *dens, *dens_prev;
static float *velocity2;


/*
  ----------------------------------------------------------------------
   free/clear/allocate simulation data
  ----------------------------------------------------------------------
*/


static void free_data(void)
{
    if (u) {
        checkCudaCall(cudaFree(u));
    }
    if (v) {
        checkCudaCall(cudaFree(v));
    }
    if (u_prev) {
        checkCudaCall(cudaFree(u_prev));
    }
    if (v_prev) {
        checkCudaCall(cudaFree(v_prev));
    }
    if (dens) {
        checkCudaCall(cudaFree(dens));
    }
    if (dens_prev) {
        checkCudaCall(cudaFree(dens_prev));
    }
    if (velocity2) {
        checkCudaCall(cudaFree(velocity2));
    }
}

static void clear_data(void)
{
    int i, size = (N + 2) * (N + 2);

    for (i = 0; i < size; i++) {
        u[i] = v[i] = u_prev[i] = v_prev[i] = dens[i] = dens_prev[i] = velocity2[i] = 0.0f;
    }
}

static int allocate_data(void)
{
    int size = (N + 2) * (N + 2);

    checkCudaCall(cudaMallocManaged(&u, size * sizeof(float)));
    checkCudaCall(cudaMallocManaged(&v, size * sizeof(float)));
    checkCudaCall(cudaMallocManaged(&u_prev, size * sizeof(float)));
    checkCudaCall(cudaMallocManaged(&v_prev, size * sizeof(float)));
    checkCudaCall(cudaMallocManaged(&dens, size * sizeof(float)));
    checkCudaCall(cudaMallocManaged(&dens_prev, size * sizeof(float)));
    checkCudaCall(cudaMallocManaged(&velocity2, size * sizeof(float)));

    return (1);
}

static void react(float* velocity2, float* d, float* u, float* v)
{
    int size = (N + 2) * (N + 2);
    float *max_velocity2;
    checkCudaCall(cudaMallocManaged(&max_velocity2, sizeof(float)));
    float *max_density;
    checkCudaCall(cudaMallocManaged(&max_density, sizeof(float)));

    launcher_get_velocity2(velocity2, N, u, v);
    void *v2_temp_storage = NULL;
    size_t v2_temp_storage_bytes = 0;
    cub::DeviceReduce::Max(v2_temp_storage, v2_temp_storage_bytes, velocity2, max_velocity2, size);
    checkCudaCall(cudaMalloc(&v2_temp_storage, v2_temp_storage_bytes));
    cub::DeviceReduce::Max(v2_temp_storage, v2_temp_storage_bytes, velocity2, max_velocity2, size);
    
    void *d_temp_storage = NULL;
    size_t d_temp_storage_bytes = 0;
    cub::DeviceReduce::Max(d_temp_storage, d_temp_storage_bytes, d, max_density, size);
    checkCudaCall(cudaMalloc(&d_temp_storage, d_temp_storage_bytes));
    cub::DeviceReduce::Max(d_temp_storage, d_temp_storage_bytes, d, max_density, size);

    checkCudaCall(cudaMemset(u, 0, size * sizeof(float)));
    checkCudaCall(cudaMemset(v, 0, size * sizeof(float)));
    checkCudaCall(cudaMemset(d, 0, size * sizeof(float)));

    if (*max_velocity2 < 0.0000005f) {
        u[IX(N / 2, N / 2)] = force * 10.0f;
        v[IX(N / 2, N / 2)] = force * 10.0f;
    }
    if (*max_density < 1.0f) {
        d[IX(N / 2, N / 2)] = source * 10.0f;
    }

    return;
}

static void dump_data(void)
{
    for (int i = 1; i < N + 1; i++) {
        for (int j = 1; j < N + 1; j++) {
            fprintf(stdout, "%f", u[IX(i,j)]);
            if (j == N) {
                fprintf(stdout, "\n");
            } else {
                fprintf(stdout, ",");
            }
        }
    }
}

static void one_step(void)
{
    react(velocity2, dens_prev, u_prev, v_prev);
    vel_step(N, u, v, u_prev, v_prev, visc, dt);
    dens_step(N, dens, dens_prev, u, v, diff, dt);
}


/*
  ----------------------------------------------------------------------
   main --- main routine
  ----------------------------------------------------------------------
*/

int main(int argc, char** argv)
{
    int i = 0;

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
        one_step();
    }
    dump_data();
    free_data();
    exit(0);
}
