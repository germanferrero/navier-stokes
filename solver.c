#include <stddef.h>

#include "solver.h"

#define IX(i, j) ((j) + (n + 2) * ((i) + (j))) 
#define IXX(i, j) ((j) + (n + 2) * (i))

#define SWAP(x0, x)      \
    {                    \
        float* tmp = x0; \
        x0 = x;          \
        x = tmp;         \
    }

typedef enum { NONE = 0,
               VERTICAL = 1,
               HORIZONTAL = 2 } boundary;

static void add_source(unsigned int n, float* x, const float* s, float dt)
{
    unsigned int size = (n + 2) * (2 * (n + 2) - 1);
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

static void lin_solve(const unsigned int n, boundary b, float *restrict x, const float *restrict x0, float a, float c)
{
    for (unsigned int k = 0; k < 20; k++) {
        for (unsigned int ix = 2; ix <= 2*n ; ix++){
            for (unsigned int jx = 1; jx < n + 1; jx++){
                x[IXX(ix, jx)] = (x0[IXX(ix, jx)] + a * (
                    x[IXX(ix - 1, jx)] +
                    x[IXX(ix + 1, jx)] +
                    x[IXX(ix - 1, jx - 1)] +
                    x[IXX(ix + 1, jx + 1)]
                )) / c;
            }
        }
        set_bnd(n, b, x);
    }
}

static void diffuse(unsigned int n, boundary b, float* x, const float* x0, float diff, float dt)
{
    float a = dt * diff * n * n;
    lin_solve(n, b, x, x0, a, 1 + 4 * a);
}

static void advect(unsigned int n, boundary b, float *restrict d, const float* d0, const float* u, const float* v, float dt)
{
    int i0, i1, j0, j1;
    float x, y, s0, t0, s1, t1;

    float dt0 = dt * n;
    for (unsigned int ix = 2; ix <= 2 * n; ix++) {
        #pragma clang loop vectorize(enable)
        for (unsigned int jx = 1; jx < n + 1; jx++) {
            unsigned int i = ix - jx;
            const unsigned int j = jx;
            x = i - dt0 * u[IXX(ix, jx)];
            y = j - dt0 * v[IXX(ix, jx)];
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
            d[IXX(ix, jx)] = s0 * (
                t0 * d0[IX(i0, j0)]
                + t1 * d0[IX(i0, j1)]
            ) +
            s1 * (
                t0 * d0[IX(i1, j0)] +
                t1 * d0[IX(i1, j1)]
            );
        }
    }
    set_bnd(n, b, d);
}

static void project(const unsigned int n, float *restrict u, float *restrict v, float *restrict p, float *restrict div)
{
    for (unsigned int ix = 2; ix <= 2 * n; ix++) {
        for (unsigned int jx = 1; jx < n + 1; jx++) {   
            div[IXX(ix, jx)] = -0.5f * (
                u[IXX(ix + 1, jx)] -
                u[IXX(ix - 1, jx)] +
                v[IXX(ix + 1, jx + 1)] -
                v[IXX(ix - 1, jx - 1)]
            ) / n;
            p[IXX(ix, jx)] = 0;
        }
    }
    set_bnd(n, NONE, div);
    set_bnd(n, NONE, p);

    lin_solve(n, NONE, p, div, 1, 4);

    for (unsigned int ix = 2; ix <= 2 * n; ix++) {
        for (unsigned int jx = 1; jx < n + 1; jx++) {
            u[IXX(ix, jx)] -= 0.5f * n * (p[IXX(ix + 1, jx)] - p[IXX(ix - 1, jx)]);
            v[IXX(ix, jx)] -= 0.5f * n * (p[IXX(ix + 1, jx + 1)] - p[IXX(ix - 1, jx - 1)]);
        }
    }
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
