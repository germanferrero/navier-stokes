#include <stddef.h>
#include <stdio.h>

#include "solver.h"
#include "indices.h"

#define IX(y,x) (rb_idx((y),(x),(n+2)))
#define IXX(y, x, stride) ((x) + (y) * (stride))

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

static void add_source(unsigned int n, float* x, const float* s, float dt)
{
    unsigned int size = (n + 2) * (n + 2);
    for (unsigned int i = 0; i < size; i++) {
        x[i] += dt * s[i];
    }
}

static void set_bnd(unsigned int n, boundary b, float* x)
{
    for (unsigned int i = 1; i <= n; i++) {
        x[IX(0, i)] = b == VERTICAL ? -x[IX(1, i)] : x[IX(1, i)];
        x[IX(n + 1, i)] = b == VERTICAL ? -x[IX(n, i)] : x[IX(n, i)];
        x[IX(i, 0)] = b == HORIZONTAL ? -x[IX(i, 1)] : x[IX(i, 1)];
        x[IX(i, n + 1)] = b == HORIZONTAL ? -x[IX(i, n)] : x[IX(i, n)];
    }
    x[IX(0, 0)] = 0.5f * (x[IX(1, 0)] + x[IX(0, 1)]);
    x[IX(0, n + 1)] = 0.5f * (x[IX(1, n + 1)] + x[IX(0, n)]);
    x[IX(n + 1, 0)] = 0.5f * (x[IX(n, 0)] + x[IX(n + 1, 1)]);
    x[IX(n + 1, n + 1)] = 0.5f * (x[IX(n, n + 1)] + x[IX(n + 1, n)]);
}

static void lin_solve_rb_step(grid_color color,
                              unsigned int n,
                              float a,
                              float c,
                              const float * restrict same0,
                              const float * restrict neigh,
                              float * restrict same)
{
    int shift = color == RED ? 1 : -1;
    unsigned int start = color == RED ? 0 : 1;

    unsigned int width = (n + 2) / 2;

    #pragma omp parallel for default(none) shared(same, same0, neigh) firstprivate(n, shift, start, width, a, c)
    for (unsigned int y = 1; y <= n; ++y) {
        const int p_shift = y % 2 == 0 ? -shift: shift;
        const int p_start = y % 2 == 0 ? 1 - start: start;
        for (unsigned int x = p_start; x < width - (1 - p_start); ++x) {
            int index = IXX(y, x, width);
            same[index] = (same0[index] + a * (neigh[index - width] +
                                               neigh[index] +
                                               neigh[index + p_shift] +
                                               neigh[index + width])) / c;
        }
    }
}

static void lin_solve(unsigned int n, boundary b,
                      float * restrict x,
                      const float * restrict x0,
                      float a, float c)
{
    unsigned int color_size = (n + 2) * ((n + 2) / 2);
    const float * red0 = x0;
    const float * blk0 = x0 + color_size;
    float * red = x;
    float * blk = x + color_size;

    for (unsigned int k = 0; k < 20; ++k) {
        lin_solve_rb_step(RED,   n, a, c, red0, blk, red);
        lin_solve_rb_step(BLACK, n, a, c, blk0, red, blk);
        set_bnd(n, b, x);
    }
}

static void diffuse(unsigned int n, boundary b, float* x, const float* x0, float diff, float dt)
{
    float a = dt * diff * n * n;
    lin_solve(n, b, x, x0, a, 1 + 4 * a);
}

static void advect_rb_step(grid_color color,
                           unsigned int n,
                           float * restrict d,
                           const float * d0,
                           const float * u,
                           const float * v,
                           float dt0)
{
    
    int i0, i1, j0, j1;
    float x, y, s0, t0, s1, t1;
    
    unsigned int start = color == RED ? 0 : 1;

    unsigned int width = (n + 2) / 2;


    #pragma omp parallel for default(none) shared(u, v, d0, d) firstprivate(n, start, width, dt0) private(i0, i1, j0, j1, x, y, s0, t0, s1, t1)
    for (unsigned int yit = 1; yit <= n; ++yit) {
        const int p_start = yit % 2 == 0 ? 1 - start: start;
        for (unsigned int xit = p_start; xit < width - (1 - p_start); ++xit) {
            int i = yit;
            int j = 1 - p_start  + 2 * xit;
            int index = IXX(yit, xit, width);
            x = i - dt0 * u[index];
            y = j - dt0 * v[index];
            if (x < 0.5f) {
                x = 0.5f;
            } else if (x > n + 0.5f) {
                x = n + 0.5f;
            }
            i0 = (int)x;
            i1 = i0 + 1;
            if (y < 0.5f) {
                y = 0.5f;
            } else if (y > n + 0.5f) {
                y = n + 0.5f;
            }
            j0 = (int)y;
            j1 = j0 + 1;
            s1 = x - i0;
            s0 = 1 - s1;
            t1 = y - j0;
            t0 = 1 - t1;
            d[index] = s0 * (t0 * d0[IX(i0, j0)] + t1 * d0[IX(i0, j1)]) + s1 * (t0 * d0[IX(i1, j0)] + t1 * d0[IX(i1, j1)]);
        }
    }
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
    advect_rb_step(RED, n, red_d, d0, red_u, red_v, dt0);
    advect_rb_step(BLACK, n, blk_d, d0, blk_u, blk_v, dt0);
    set_bnd(n, b, d);
}

static void project_before_rb_step(grid_color color,
                              unsigned int n,
                              float * restrict div,
                              const float * u,
                              const float * v,
                              float * restrict p)
{
    int shift = color == RED ? 1 : -1;
    unsigned int start = color == RED ? 0 : 1;

    unsigned int width = (n + 2) / 2;

    #pragma omp parallel for default(none) shared(u, v, div, p) firstprivate(n, shift, start, width)
    for (unsigned int y = 1; y <= n; ++y) {
        const int p_shift = y % 2 == 0 ? -shift: shift;
        const int p_start = y % 2 == 0 ? 1 - start: start;
        for (unsigned int x = p_start; x < width - (1 - p_start); ++x) {
            int index = IXX(y, x, width);
            div[index] = -0.5f * (u[index + width] -
                                  u[index - width] +
                                  (p_shift * v[index + p_shift]) +
                                  (-p_shift * v[index])) / n;
            p[index] = 0;
        }
    } 
}

static void project_after_rb_step(grid_color color,
                              unsigned int n,
                              float * restrict u,
                              float * restrict v,
                              const float * p)
{
    int shift = color == RED ? 1 : -1;
    unsigned int start = color == RED ? 0 : 1;

    unsigned int width = (n + 2) / 2;

    #pragma omp parallel for default(none) shared(u, v, p) firstprivate(n, shift, start, width)
    for (unsigned int y = 1; y <= n; ++y) {
        const int p_shift = y % 2 == 0 ? -shift: shift;
        const int p_start = y % 2 == 0 ? 1 - start: start;
        for (unsigned int x = p_start; x < width - (1 - p_start); ++x) {
            int index = IXX(y, x, width);
            u[index] -= 0.5f * n * (p[index + width] - p[index - width]);
            v[index] -= 0.5f * n * ((p_shift * p[index + p_shift]) + (-p_shift * p[index]));
        }
    }
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
    project_before_rb_step(RED, n, red_div, blk_u, blk_v, red_p);
    project_before_rb_step(BLACK, n, blk_div, red_u, red_v, blk_p);
    set_bnd(n, NONE, div);
    set_bnd(n, NONE, p);

    lin_solve(n, NONE, p, div, 1, 4);

    project_after_rb_step(RED, n, red_u, red_v, blk_p);
    project_after_rb_step(BLACK, n, blk_u, blk_v, red_p);
    set_bnd(n, VERTICAL, u);
    set_bnd(n, HORIZONTAL, v);
}

void dens_step(unsigned int n, float* x, float* x0, float* u, float* v, float diff, float dt)
{
    add_source(n, x, x0, dt);
    SWAP(x0, x);
    diffuse(n, NONE, x, x0, diff, dt);
    SWAP(x0, x);
    advect(n, NONE, x, x0, u, v, dt);
}

void vel_step(unsigned int n, float* u, float* v, float* u0, float* v0, float visc, float dt)
{
    add_source(n, u, u0, dt);
    add_source(n, v, v0, dt);
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
