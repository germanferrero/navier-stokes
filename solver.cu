#include <stddef.h>
#include <stdio.h>
#include <cuda_runtime.h>

#include "cuda_helper.h"
#include "solver.h"
#include "indices.h"

#define IX(y,x) (rb_idx((y),(x),(n+2)))
#define IXX(y, x, stride) ((x) + (y) * (stride))

#define BLOCK_SIZE 16

#define SWAP(x0, x)      \
    {                    \
        float* tmp = x0; \
        x0 = x;          \
        x = tmp;         \
    }

typedef enum { NONE = 0,
               VERTICAL = 1,
               HORIZONTAL = 2 } boundary;

typedef enum { RED, BLACK } grid_color;

template <typename T>
T div_ceil(T a, T b) {
    return (a + b - 1) / b;
}

__global__ void kernel_add_forces(float *max_velocity2, unsigned int n, float force, float * u, float * v){
    if (*max_velocity2 < 0.0000005f) {
        u[IX(n / 2, n / 2)] = force * 10.0f;
        v[IX(n / 2, n / 2)] = force * 10.0f;
    }
}

void launcher_add_forces(float *max_velocity2, unsigned int n, float force, float * u, float * v){
    kernel_add_forces<<<1,1>>>(max_velocity2, n, force, u, v);
    checkCudaCall(cudaGetLastError());
    checkCudaCall(cudaDeviceSynchronize());
}

__global__ void kernel_add_densities(float *max_density, unsigned int n, float source, float * d){
    if (*max_density < 1.0f) {
        d[IX(n / 2, n / 2)] = source * 10.0f;
    }
}


void launcher_add_densities(float *max_density, unsigned int n, float source, float * d){
    kernel_add_densities<<<1,1>>>(max_density, n, source, d);
    checkCudaCall(cudaGetLastError());
    checkCudaCall(cudaDeviceSynchronize());
}


__global__ void kernel_get_velocity2(float * velocity2, unsigned int n, const float* u, const float* v) {
    unsigned int width = (n + 2) / 2;
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x > (n + 1) || y > (n + 1)){
        return;
    }
    int index = IXX(y, x, width);
    velocity2[index] = u[index] * u[index] + v[index] * v[index];
}

void launcher_get_velocity2(float * velocity2, unsigned int n, const float* u, const float* v) {
    unsigned int width = (n + 2) / 2;
    unsigned int N_BLOCKS_X = div_ceil(width, (uint) BLOCK_SIZE);
    unsigned int N_BLOCKS_Y = div_ceil(2 * (n + 2), (uint) BLOCK_SIZE);
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid(N_BLOCKS_X, N_BLOCKS_Y);
    kernel_get_velocity2<<<grid, block>>>(velocity2, n, u, v);
    checkCudaCall(cudaGetLastError());
    checkCudaCall(cudaDeviceSynchronize());
}

__global__ void kernel_linsolve_rb_step(grid_color color,
                                        unsigned int n,
                                        float a,
                                        float c,
                                        const float * same0,
                                        const float * neigh,
                                        float * same
)
{
    unsigned int width = (n + 2) / 2;
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int index = IXX(y, x, width);
    
    int start_first_row = color == RED ? 0 : 1;
    int start = y % 2 == 1 ? start_first_row : 1 - start_first_row;
    if (x < start) {
        return;
    }
    if (x >= width + start - 1) {
        return;
    }
    if (y == 0 || y >= n+1) {
        return;
    }
    
    int shift_first_row = color == RED ? 1 : -1;
    int shift = y % 2 == 1 ? shift_first_row: -shift_first_row;
    same[index] = (same0[index] + a * (
        neigh[index - width] +
        neigh[index] +
        neigh[index + shift] +
        neigh[index + width]
    )) / c;
}

void launcher_linsolve_rb_step(grid_color color,
                              unsigned int n,
                              float a,
                              float c,
                              const float * same0,
                              const float * neigh,
                              float * same)
{
    unsigned int width = (n + 2) / 2;
    unsigned int N_BLOCKS_X = div_ceil(width, (uint) BLOCK_SIZE);
    unsigned int N_BLOCKS_Y = div_ceil(n + 2, (uint) BLOCK_SIZE);
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid(N_BLOCKS_X, N_BLOCKS_Y);
    kernel_linsolve_rb_step<<<grid, block>>>(color, n, a, c, same0, neigh, same);
    checkCudaCall(cudaGetLastError());
    checkCudaCall(cudaDeviceSynchronize());
}

__global__ void kernel_add_source(unsigned int n, float* m, const float* s, float dt)
{
    unsigned int width = (n + 2) / 2;
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x > (n + 1) || y > (n + 1)){
        return;
    }
    int index = IXX(y, x, width);
    m[index] += dt * s[index];
}

void launcher_add_source(unsigned int n, float* x, const float* s, float dt)
{
    unsigned int width = (n + 2) / 2;
    unsigned int N_BLOCKS_X = div_ceil(width, (uint) BLOCK_SIZE);
    unsigned int N_BLOCKS_Y = div_ceil(2 * (n + 2), (uint) BLOCK_SIZE);
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid(N_BLOCKS_X, N_BLOCKS_Y);
    kernel_add_source<<<grid, block>>>(n, x, s, dt);
}


__global__ void kernel_set_bnd(unsigned int n, boundary b, float* m)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x > (n + 1) || y > (n + 1)){
        return;
    }
    
    int index = IX(x, y);

    if (x == 0){
        if (y == 0){
            m[index] = 0.5f * (m[IX(1, 0)] + m[IX(0, 1)]);
        }
        if (y == n+1){
            m[index] = 0.5f * (m[IX(1, n + 1)] + m[IX(0, n)]);
        }
        else {
            m[index] = b == VERTICAL ? -m[IX(1, y)] : m[IX(1, y)];
        }
    }
    else if (x == n+1) {
        if (y == 0){
            m[index] = 0.5f * (m[IX(n, 0)] + m[IX(n + 1, 1)]);
        }
        if (y == n+1){
            m[index] = 0.5f * (m[IX(n, n + 1)] + m[IX(n + 1, n)]);
        }
        else {
            m[index] = b == VERTICAL ? -m[IX(n, y)] : m[IX(n, y)];
        }
    }
    else if (y == 0) {
        m[index] = b == HORIZONTAL ? -m[IX(x, 1)] : m[IX(x, 1)];
    }
    else if (y == n+1) {
        m[index] = b == HORIZONTAL ? -m[IX(x, n)] : m[IX(x, n)];
    }
}

void launcher_set_bnd(unsigned int n, boundary b, float* x)
{
    unsigned int N_BLOCKS = div_ceil(n + 2, (uint) BLOCK_SIZE);
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid(N_BLOCKS, N_BLOCKS);
    kernel_set_bnd<<<grid, block>>>(n, b, x);
    checkCudaCall(cudaGetLastError());
    checkCudaCall(cudaDeviceSynchronize());
}

static void lin_solve(unsigned int n, boundary b,
                      float * x,
                      const float * x0,
                      float a, float c)
{
    unsigned int color_size = (n + 2) * ((n + 2) / 2);
    const float * red0 = x0;
    const float * blk0 = x0 + color_size;
    float * red = x;
    float * blk = x + color_size;
    
    for (unsigned int k = 0; k < 20; ++k) {
        launcher_linsolve_rb_step(RED, n, a, c, red0, blk, red);
        launcher_linsolve_rb_step(BLACK, n, a, c, blk0, red, blk);
        launcher_set_bnd(n, b, x);
    }
}

static void diffuse(unsigned int n, boundary b, float* x, const float* x0, float diff, float dt)
{
    float a = dt * diff * n * n;
    lin_solve(n, b, x, x0, a, 1 + 4 * a);
}

__global__ void kernel_advect_rb_step(
    grid_color color,
    unsigned int n,
    float * d,
    const float * d0,
    const float * u,
    const float * v,
    float dt0)
{
    unsigned int width = (n + 2) / 2;
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int index = IXX(y, x, width);
    
    int start_first_row = color == RED ? 0 : 1;
    int start = y % 2 == 1 ? start_first_row : 1 - start_first_row;
    if (x < start) {
        return;
    }
    if (x >= width + start - 1) {
        return;
    }
    if (y == 0 || y >= n+1) {
        return;
    }
    
    int i0, i1, j0, j1;
    float x_, y_, s0, t0, s1, t1;

    int i = y;
    int j = 1 - start + 2 * x;
    x_ = i - dt0 * u[index];
    y_ = j - dt0 * v[index];
    if (x_ < 0.5f) {
        x_ = 0.5f;
    } else if (x_ > n + 0.5f) {
        x_ = n + 0.5f;
    }
    i0 = (int)x_;
    i1 = i0 + 1;
    if (y_ < 0.5f) {
        y_ = 0.5f;
    } else if (y_ > n + 0.5f) {
        y_ = n + 0.5f;
    }
    j0 = (int)y_;
    j1 = j0 + 1;
    s1 = x_ - i0;
    s0 = 1 - s1;
    t1 = y_ - j0;
    t0 = 1 - t1;
    d[index] = s0 * (t0 * d0[IX(i0, j0)] + t1 * d0[IX(i0, j1)]) + s1 * (t0 * d0[IX(i1, j0)] + t1 * d0[IX(i1, j1)]);
}

void launcher_advect_rb_step(
    grid_color color,
    unsigned int n,
    float * d,
    const float * d0,
    const float * u,
    const float * v,
    float dt0)
{
    unsigned int width = (n + 2) / 2;
    unsigned int N_BLOCKS_X = div_ceil(width, (uint) BLOCK_SIZE);
    unsigned int N_BLOCKS_Y = div_ceil(n + 2, (uint) BLOCK_SIZE);
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid(N_BLOCKS_X, N_BLOCKS_Y);
    kernel_advect_rb_step<<<grid, block>>>(color, n, d, d0, u, v, dt0);
    checkCudaCall(cudaGetLastError());
    checkCudaCall(cudaDeviceSynchronize());
}

static void advect(unsigned int n, boundary b, float* d, const float* d0, const float* u, const float* v, float dt)
{
    unsigned int color_size = (n + 2) * ((n + 2) / 2);
    float * red_d = d;
    float * blk_d = d + color_size;
    const float * red_u = u;
    const float * blk_u = u + color_size;
    const float * red_v = v;
    const float * blk_v = v + color_size;
    float dt0 = dt * n;
    launcher_advect_rb_step(RED, n, red_d, d0, red_u, red_v, dt0);
    launcher_advect_rb_step(BLACK, n, blk_d, d0, blk_u, blk_v, dt0);
    launcher_set_bnd(n, b, d);
}

__global__ void kernel_project_before_rb_step(
    grid_color color,
    unsigned int n,
    float * div,
    const float * u,
    const float * v,
    float * p
)
{
    unsigned int width = (n + 2) / 2;
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int index = IXX(y, x, width);

    int start_first_row = color == RED ? 0 : 1;
    int start = y % 2 == 1 ? start_first_row : 1 - start_first_row;
    if (x < start) {
        return;
    }
    if (x >= width + start - 1) {
        return;
    }
    if (y == 0 || y >= n+1) {
        return;
    }

    int shift_first_row = color == RED ? 1 : -1;
    int shift = y % 2 == 1 ? shift_first_row: -shift_first_row;
    div[index] = -0.5f * (u[index + width] -
                          u[index - width] +
                          (shift * v[index + shift]) +
                          (-shift * v[index])) / n;
    p[index] = 0;
}

void launcher_project_before_rb_step(
    grid_color color,
    unsigned int n,
    float * div,
    const float * u,
    const float * v,
    float * p
)
{
    unsigned int width = (n + 2) / 2;
    unsigned int N_BLOCKS_X = div_ceil(width, (uint) BLOCK_SIZE);
    unsigned int N_BLOCKS_Y = div_ceil(n + 2, (uint) BLOCK_SIZE);
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid(N_BLOCKS_X, N_BLOCKS_Y);
    kernel_project_before_rb_step<<<grid, block>>>(color, n, div, u, v, p);
    checkCudaCall(cudaGetLastError());
    checkCudaCall(cudaDeviceSynchronize());
}

__global__ void kernel_project_after_rb_step(
    grid_color color,
    unsigned int n,
    float * u,
    float * v,
    const float * p)
{
    unsigned int width = (n + 2) / 2;
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int index = IXX(y, x, width);

    int start_first_row = color == RED ? 0 : 1;
    int start = y % 2 == 1 ? start_first_row : 1 - start_first_row;
    if (x < start) {
        return;
    }
    if (x >= width + start - 1) {
        return;
    }
    if (y == 0 || y >= n+1) {
        return;
    }

    int shift_first_row = color == RED ? 1 : -1;
    int shift = y % 2 == 1 ? shift_first_row: -shift_first_row;
    u[index] -= 0.5f * n * (p[index + width] - p[index - width]);
    v[index] -= 0.5f * n * ((shift * p[index + shift]) + (-shift * p[index]));
}

void launcher_project_after_rb_step(
    grid_color color,
    unsigned int n,
    float * u,
    float * v,
    const float * p
)
{
    unsigned int width = (n + 2) / 2;
    unsigned int N_BLOCKS_X = div_ceil(width, (uint) BLOCK_SIZE);
    unsigned int N_BLOCKS_Y = div_ceil(n + 2, (uint) BLOCK_SIZE);
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid(N_BLOCKS_X, N_BLOCKS_Y);
    kernel_project_after_rb_step<<<grid, block>>>(color, n, u, v, p);
    checkCudaCall(cudaGetLastError());
    checkCudaCall(cudaDeviceSynchronize());
}

static void project(unsigned int n, float* u, float* v, float* p, float* div)
{
    unsigned int color_size = (n + 2) * ((n + 2) / 2);
    float * red_v = v;
    float * blk_v = v + color_size;
    float * red_u = u;
    float * blk_u = u + color_size;
    float * red_div = div;
    float * blk_div = div + color_size;
    float * red_p = p;
    float * blk_p = p + color_size;
    launcher_project_before_rb_step(RED, n, red_div, blk_u, blk_v, red_p);
    launcher_project_before_rb_step(BLACK, n, blk_div, red_u, red_v, blk_p);
    launcher_set_bnd(n, NONE, div);
    launcher_set_bnd(n, NONE, p);

    lin_solve(n, NONE, p, div, 1, 4);

    launcher_project_after_rb_step(RED, n, red_u, red_v, blk_p);
    launcher_project_after_rb_step(BLACK, n, blk_u, blk_v, red_p);
    launcher_set_bnd(n, VERTICAL, u);
    launcher_set_bnd(n, HORIZONTAL, v);
}

void dens_step(unsigned int n, float* x, float* x0, float* u, float* v, float diff, float dt)
{
    launcher_add_source(n, x, x0, dt);
    SWAP(x0, x);
    diffuse(n, NONE, x, x0, diff, dt);
    SWAP(x0, x);
    advect(n, NONE, x, x0, u, v, dt);
}

void vel_step(unsigned int n, float* u, float* v, float* u0, float* v0, float visc, float dt)
{
    launcher_add_source(n, u, u0, dt);
    launcher_add_source(n, v, v0, dt);
    SWAP(u0, u);
    diffuse(n, VERTICAL, u, u0, visc, dt);
    SWAP(v0, v);
    diffuse(n, HORIZONTAL, v, v0, visc, dt);
    project(n, u, v, u0, v0);
    SWAP(u0, u);
    SWAP(v0, v);
    advect(n, VERTICAL, u, u0, u0, v0, dt);
    advect(n, HORIZONTAL, v, v0, u0, v0, dt);
    project(n, u, v, u0, v0);
}
