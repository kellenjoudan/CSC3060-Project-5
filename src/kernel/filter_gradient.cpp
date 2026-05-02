#include "filter_gradient.h"

#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <random>

void initialize_filter_gradient(filter_gradient_args* args,
                        std::size_t width,
                        std::size_t height,
                        std::uint_fast64_t seed) {
    if (!args) {
        return;
    }

    assert(width >= 3);
    assert(height >= 3);

    args->width = width;
    args->height = height;
    args->out = 0.0f;

    const std::size_t count = width * height;

    std::mt19937_64 gen(seed);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    args->data.a.resize(count);
    args->data.b.resize(count);
    args->data.c.resize(count);
    args->data.d.resize(count);
    args->data.e.resize(count);
    args->data.f.resize(count);
    args->data.g.resize(count);
    args->data.h.resize(count);
    args->data.i.resize(count);

    for (std::size_t k = 0; k < count; ++k) {
        args->data.a[k] = dist(gen);
        args->data.b[k] = dist(gen);
        args->data.c[k] = dist(gen);
        args->data.d[k] = dist(gen);
        args->data.e[k] = dist(gen);
        args->data.f[k] = dist(gen);
        args->data.g[k] = dist(gen);
        args->data.h[k] = dist(gen);
        args->data.i[k] = dist(gen);
    }
}

void naive_filter_gradient(float& out, const data_struct& data,
                   std::size_t width, std::size_t height) {
    const std::size_t W = width;
    const std::size_t H = height;
    constexpr float inv9 = 1.0f / 9.0f;

    double total = 0.0f;

    for (std::size_t y = 1; y + 1 < H; ++y) {
        for (std::size_t x = 1; x + 1 < W; ++x) {

            double sum_a = 0.0, sum_b = 0.0, sum_c = 0.0;
            for (int dy = -1; dy <= 1; ++dy) {
                const std::size_t row = (y + dy) * W;
                for (int dx = -1; dx <= 1; ++dx) {
                    const std::size_t idx = row + (x + dx);
                    sum_a += data.a[idx];
                    sum_b += data.b[idx];
                    sum_c += data.c[idx];
                }
            }
            const float avg_a = sum_a * inv9;
            const float avg_b = sum_b * inv9;
            const float avg_c = sum_c * inv9;
            const float p1 = avg_a * avg_b + avg_c;

            const std::size_t ym1 = (y - 1) * W;
            const std::size_t y0  = y * W;
            const std::size_t yp1 = (y + 1) * W;

            const std::size_t xm1 = x - 1;
            const std::size_t x0  = x;
            const std::size_t xp1 = x + 1;

            const float sobel_dx =
                -data.d[ym1 + xm1] + data.d[ym1 + xp1]
                -2.0f * data.d[y0 + xm1] + 2.0f * data.d[y0 + xp1]
                -data.d[yp1 + xm1] + data.d[yp1 + xp1];

            const float sobel_ex =
                -data.e[ym1 + xm1] + data.e[ym1 + xp1]
                -2.0f * data.e[y0 + xm1] + 2.0f * data.e[y0 + xp1]
                -data.e[yp1 + xm1] + data.e[yp1 + xp1];

            const float sobel_fx =
                -data.f[ym1 + xm1] + data.f[ym1 + xp1]
                -2.0f * data.f[y0 + xm1] + 2.0f * data.f[y0 + xp1]
                -data.f[yp1 + xm1] + data.f[yp1 + xp1];

            const float p2 = sobel_dx * sobel_ex + sobel_fx;

            const float sobel_gy =
                -data.g[ym1 + xm1] - 2.0f * data.g[ym1 + x0] - data.g[ym1 + xp1]
                + data.g[yp1 + xm1] + 2.0f * data.g[yp1 + x0] + data.g[yp1 + xp1];

            const float sobel_hy =
                -data.h[ym1 + xm1] - 2.0f * data.h[ym1 + x0] - data.h[ym1 + xp1]
                + data.h[yp1 + xm1] + 2.0f * data.h[yp1 + x0] + data.h[yp1 + xp1];

            const float sobel_iy =
                -data.i[ym1 + xm1] - 2.0f * data.i[ym1 + x0] - data.i[ym1 + xp1]
                + data.i[yp1 + xm1] + 2.0f * data.i[yp1 + x0] + data.i[yp1 + xp1];

            const float p3 = sobel_gy * sobel_hy + sobel_iy;

            total += p1 + p2 + p3;
        }
    }

    out = total;
}

// TODO: You may need to add a function to convert data structure (not 
// included in time measurement), then implement your version in 
// stu_filter_gradient, whch is called by stu_filter_gradient_wrapper.
void stu_filter_gradient(float& out, const data_struct& data,
                        std::size_t width, std::size_t height) {

    const size_t W = width;
    const size_t H = height;
    constexpr float inv9 = 1.0f / 9.0f;
    constexpr float inv81 = inv9 * inv9;  // precompute (1/9)*(1/9)

    double total = 0.0;

    for (size_t y = 1; y + 1 < H; ++y) {
        const size_t y_minus_1 = (y - 1) * W;
        const size_t y_curr = y * W;
        const size_t y_plus_1 = (y + 1) * W;

        const float* a0 = &data.a[y_minus_1];
        const float* a1 = &data.a[y_curr];
        const float* a2 = &data.a[y_plus_1];

        const float* b0 = &data.b[y_minus_1];
        const float* b1 = &data.b[y_curr];
        const float* b2 = &data.b[y_plus_1];

        const float* c0 = &data.c[y_minus_1];
        const float* c1 = &data.c[y_curr];
        const float* c2 = &data.c[y_plus_1];

        const float* d0 = &data.d[y_minus_1];
        const float* d1 = &data.d[y_curr];
        const float* d2 = &data.d[y_plus_1];

        const float* e0 = &data.e[y_minus_1];
        const float* e1 = &data.e[y_curr];
        const float* e2 = &data.e[y_plus_1];

        const float* f0 = &data.f[y_minus_1];
        const float* f1 = &data.f[y_curr];
        const float* f2 = &data.f[y_plus_1];

        const float* g0 = &data.g[y_minus_1];
        const float* g2 = &data.g[y_plus_1];

        const float* h0 = &data.h[y_minus_1];
        const float* h2 = &data.h[y_plus_1];

        const float* i0 = &data.i[y_minus_1];
        const float* i2 = &data.i[y_plus_1];

        size_t x = 1;

        for (; x + 2 < W; x += 2) {
            // Precompute indices
            const size_t xm1 = x - 1;
            const size_t xp1 = x + 1;
            const size_t x2 = x + 1;      // second pixel center
            const size_t x2m1 = x;         // x2 - 1
            const size_t x2p1 = x + 2;     // x2 + 1

            // Sum A for pixel 1
            float sum_a0 = (a0[xm1] + a0[x] + a0[xp1]) +
                          (a1[xm1] + a1[x] + a1[xp1]) +
                          (a2[xm1] + a2[x] + a2[xp1]);

            // Sum B for pixel 1
            float sum_b0 = (b0[xm1] + b0[x] + b0[xp1]) +
                          (b1[xm1] + b1[x] + b1[xp1]) +
                          (b2[xm1] + b2[x] + b2[xp1]);

            // Sum C for pixel 1
            float sum_c0 = (c0[xm1] + c0[x] + c0[xp1]) +
                          (c1[xm1] + c1[x] + c1[xp1]) +
                          (c2[xm1] + c2[x] + c2[xp1]);

            float p1_0 = sum_a0 * sum_b0 * inv81 + sum_c0 * inv9;

            // DX gradients for pixel 1
            float dx_d0 = -d0[xm1] + d0[xp1] - 2.0f*d1[xm1] + 2.0f*d1[xp1] - d2[xm1] + d2[xp1];
            float dx_e0 = -e0[xm1] + e0[xp1] - 2.0f*e1[xm1] + 2.0f*e1[xp1] - e2[xm1] + e2[xp1];
            float dx_f0 = -f0[xm1] + f0[xp1] - 2.0f*f1[xm1] + 2.0f*f1[xp1] - f2[xm1] + f2[xp1];

            float p2_0 = dx_d0 * dx_e0 + dx_f0;

            // DY gradients for pixel 1
            float dy_g0 = -g0[xm1] - 2.0f*g0[x] - g0[xp1] + g2[xm1] + 2.0f*g2[x] + g2[xp1];
            float dy_h0 = -h0[xm1] - 2.0f*h0[x] - h0[xp1] + h2[xm1] + 2.0f*h2[x] + h2[xp1];
            float dy_i0 = -i0[xm1] - 2.0f*i0[x] - i0[xp1] + i2[xm1] + 2.0f*i2[x] + i2[xp1];

            float p3_0 = dy_g0 * dy_h0 + dy_i0;

            // Sum A for pixel 2
            float sum_a1 = (a0[x2m1] + a0[x2] + a0[x2p1]) +
                          (a1[x2m1] + a1[x2] + a1[x2p1]) +
                          (a2[x2m1] + a2[x2] + a2[x2p1]);

            // Sum B for pixel 2
            float sum_b1 = (b0[x2m1] + b0[x2] + b0[x2p1]) +
                          (b1[x2m1] + b1[x2] + b1[x2p1]) +
                          (b2[x2m1] + b2[x2] + b2[x2p1]);

            // Sum C for pixel 2
            float sum_c1 = (c0[x2m1] + c0[x2] + c0[x2p1]) +
                          (c1[x2m1] + c1[x2] + c1[x2p1]) +
                          (c2[x2m1] + c2[x2] + c2[x2p1]);

            float p1_1 = sum_a1 * sum_b1 * inv81 + sum_c1 * inv9;

            // DX gradients for pixel 2
            float dx_d1 = -d0[x2m1] + d0[x2p1] - 2.0f*d1[x2m1] + 2.0f*d1[x2p1] - d2[x2m1] + d2[x2p1];
            float dx_e1 = -e0[x2m1] + e0[x2p1] - 2.0f*e1[x2m1] + 2.0f*e1[x2p1] - e2[x2m1] + e2[x2p1];
            float dx_f1 = -f0[x2m1] + f0[x2p1] - 2.0f*f1[x2m1] + 2.0f*f1[x2p1] - f2[x2m1] + f2[x2p1];

            float p2_1 = dx_d1 * dx_e1 + dx_f1;

            // DY gradients for pixel 2
            float dy_g1 = -g0[x2m1] - 2.0f*g0[x2] - g0[x2p1] + g2[x2m1] + 2.0f*g2[x2] + g2[x2p1];
            float dy_h1 = -h0[x2m1] - 2.0f*h0[x2] - h0[x2p1] + h2[x2m1] + 2.0f*h2[x2] + h2[x2p1];
            float dy_i1 = -i0[x2m1] - 2.0f*i0[x2] - i0[x2p1] + i2[x2m1] + 2.0f*i2[x2] + i2[x2p1];

            float p3_1 = dy_g1 * dy_h1 + dy_i1;

            total += p1_0 + p2_0 + p3_0 + p1_1 + p2_1 + p3_1;
        }

        // Tail loop for remaining pixels (when width is odd)
        for (; x + 1 < W; ++x) {
            float sum_a = (a0[x-1] + a0[x] + a0[x+1]) +
                         (a1[x-1] + a1[x] + a1[x+1]) +
                         (a2[x-1] + a2[x] + a2[x+1]);

            float sum_b = (b0[x-1] + b0[x] + b0[x+1]) +
                         (b1[x-1] + b1[x] + b1[x+1]) +
                         (b2[x-1] + b2[x] + b2[x+1]);

            float sum_c = (c0[x-1] + c0[x] + c0[x+1]) +
                         (c1[x-1] + c1[x] + c1[x+1]) +
                         (c2[x-1] + c2[x] + c2[x+1]);

            total += sum_a * sum_b * inv81 + sum_c * inv9;
        }
    }

    out = static_cast<float>(total);
}

void naive_filter_gradient_wrapper(void* ctx) {
    auto& args = *static_cast<filter_gradient_args*>(ctx);
    args.out = 0.0f;
    naive_filter_gradient(args.out, args.data, args.width, args.height);
}
void stu_filter_gradient_wrapper(void* ctx) {
    auto& args = *static_cast<filter_gradient_args*>(ctx);
    args.out = 0.0f;
    stu_filter_gradient(args.out, args.data, args.width, args.height);
}

bool filter_gradient_check(void* stu_ctx, void* ref_ctx, lab_test_func naive_func) {
    auto& stu_args = *static_cast<filter_gradient_args*>(stu_ctx);
    auto& ref_args = *static_cast<filter_gradient_args*>(ref_ctx);

    ref_args.out = 0.0f;
    naive_func(ref_ctx);

    const auto eps = ref_args.epsilon;
    const double s = static_cast<double>(stu_args.out);
    const double r = static_cast<double>(ref_args.out);
    const double err = std::abs(s - r);
    const double atol = 1e-6;
    const double rel = (std::abs(r) > atol) ? err / std::abs(r) : err;
    debug_log("DEBUG: filter_gradient stu={} ref={} err={} rel={}\n",
              stu_args.out,
              ref_args.out,
              err,
              rel);

    return err <= (atol + eps * std::abs(r));
}
