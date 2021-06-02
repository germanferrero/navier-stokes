#pragma once
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"

static inline size_t rb_idx(size_t x, size_t y, size_t dim) {
    size_t base = ((x % 2) ^ (y % 2)) * dim * (dim / 2);
    size_t offset = (x / 2) + y * (dim / 2);
    return base + offset;
}

static size_t idx(size_t x, size_t y, size_t stride) {
    return x + y * stride;
}

#pragma GCC diagnostic pop
