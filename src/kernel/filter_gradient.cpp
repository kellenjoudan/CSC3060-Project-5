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
    constexpr float inv81 = inv9 * inv9;
    constexpr float two = 2.0f;

    // Use float for better SIMD vectorization
    float total = 0.0f;

    // Get raw pointers once
    const float* const a_ptr = data.a.data();
    const float* const b_ptr = data.b.data();
    const float* const c_ptr = data.c.data();
    const float* const d_ptr = data.d.data();
    const float* const e_ptr = data.e.data();
    const float* const f_ptr = data.f.data();
    const float* const g_ptr = data.g.data();
    const float* const h_ptr = data.h.data();
    const float* const i_ptr = data.i.data();

    for (size_t y = 1; y + 1 < H; ++y) {
        const size_t y_minus_1 = (y - 1) * W;
        const size_t y_curr = y * W;
        const size_t y_plus_1 = (y + 1) * W;

        // Precompute all row pointers
        const float* const a0 = a_ptr + y_minus_1;
        const float* const a1 = a_ptr + y_curr;
        const float* const a2 = a_ptr + y_plus_1;
        
        const float* const b0 = b_ptr + y_minus_1;
        const float* const b1 = b_ptr + y_curr;
        const float* const b2 = b_ptr + y_plus_1;
        
        const float* const c0 = c_ptr + y_minus_1;
        const float* const c1 = c_ptr + y_curr;
        const float* const c2 = c_ptr + y_plus_1;
        
        const float* const d0 = d_ptr + y_minus_1;
        const float* const d1 = d_ptr + y_curr;
        const float* const d2 = d_ptr + y_plus_1;
        
        const float* const e0 = e_ptr + y_minus_1;
        const float* const e1 = e_ptr + y_curr;
        const float* const e2 = e_ptr + y_plus_1;
        
        const float* const f0 = f_ptr + y_minus_1;
        const float* const f1 = f_ptr + y_curr;
        const float* const f2 = f_ptr + y_plus_1;
        
        const float* const g0 = g_ptr + y_minus_1;
        const float* const g2 = g_ptr + y_plus_1;
        
        const float* const h0 = h_ptr + y_minus_1;
        const float* const h2 = h_ptr + y_plus_1;
        
        const float* const i0 = i_ptr + y_minus_1;
        const float* const i2 = i_ptr + y_plus_1;

        // Process 8 pixels at a time for maximum ILP
        size_t x = 1;
        const size_t x_end = W - 1;
        
        for (; x + 7 < x_end; x += 8) {
            // Pixel 1
            const float s1a = a0[x-1] + a0[x] + a0[x+1] + a1[x-1] + a1[x] + a1[x+1] + a2[x-1] + a2[x] + a2[x+1];
            const float s1b = b0[x-1] + b0[x] + b0[x+1] + b1[x-1] + b1[x] + b1[x+1] + b2[x-1] + b2[x] + b2[x+1];
            const float s1c = c0[x-1] + c0[x] + c0[x+1] + c1[x-1] + c1[x] + c1[x+1] + c2[x-1] + c2[x] + c2[x+1];
            
            // Pixel 2
            const float s2a = a0[x] + a0[x+1] + a0[x+2] + a1[x] + a1[x+1] + a1[x+2] + a2[x] + a2[x+1] + a2[x+2];
            const float s2b = b0[x] + b0[x+1] + b0[x+2] + b1[x] + b1[x+1] + b1[x+2] + b2[x] + b2[x+1] + b2[x+2];
            const float s2c = c0[x] + c0[x+1] + c0[x+2] + c1[x] + c1[x+1] + c1[x+2] + c2[x] + c2[x+1] + c2[x+2];
            
            // Pixel 3
            const float s3a = a0[x+1] + a0[x+2] + a0[x+3] + a1[x+1] + a1[x+2] + a1[x+3] + a2[x+1] + a2[x+2] + a2[x+3];
            const float s3b = b0[x+1] + b0[x+2] + b0[x+3] + b1[x+1] + b1[x+2] + b1[x+3] + b2[x+1] + b2[x+2] + b2[x+3];
            const float s3c = c0[x+1] + c0[x+2] + c0[x+3] + c1[x+1] + c1[x+2] + c1[x+3] + c2[x+1] + c2[x+2] + c2[x+3];
            
            // Pixel 4
            const float s4a = a0[x+2] + a0[x+3] + a0[x+4] + a1[x+2] + a1[x+3] + a1[x+4] + a2[x+2] + a2[x+3] + a2[x+4];
            const float s4b = b0[x+2] + b0[x+3] + b0[x+4] + b1[x+2] + b1[x+3] + b1[x+4] + b2[x+2] + b2[x+3] + b2[x+4];
            const float s4c = c0[x+2] + c0[x+3] + c0[x+4] + c1[x+2] + c1[x+3] + c1[x+4] + c2[x+2] + c2[x+3] + c2[x+4];
            
            // Pixel 5
            const float s5a = a0[x+3] + a0[x+4] + a0[x+5] + a1[x+3] + a1[x+4] + a1[x+5] + a2[x+3] + a2[x+4] + a2[x+5];
            const float s5b = b0[x+3] + b0[x+4] + b0[x+5] + b1[x+3] + b1[x+4] + b1[x+5] + b2[x+3] + b2[x+4] + b2[x+5];
            const float s5c = c0[x+3] + c0[x+4] + c0[x+5] + c1[x+3] + c1[x+4] + c1[x+5] + c2[x+3] + c2[x+4] + c2[x+5];
            
            // Pixel 6
            const float s6a = a0[x+4] + a0[x+5] + a0[x+6] + a1[x+4] + a1[x+5] + a1[x+6] + a2[x+4] + a2[x+5] + a2[x+6];
            const float s6b = b0[x+4] + b0[x+5] + b0[x+6] + b1[x+4] + b1[x+5] + b1[x+6] + b2[x+4] + b2[x+5] + b2[x+6];
            const float s6c = c0[x+4] + c0[x+5] + c0[x+6] + c1[x+4] + c1[x+5] + c1[x+6] + c2[x+4] + c2[x+5] + c2[x+6];
            
            // Pixel 7
            const float s7a = a0[x+5] + a0[x+6] + a0[x+7] + a1[x+5] + a1[x+6] + a1[x+7] + a2[x+5] + a2[x+6] + a2[x+7];
            const float s7b = b0[x+5] + b0[x+6] + b0[x+7] + b1[x+5] + b1[x+6] + b1[x+7] + b2[x+5] + b2[x+6] + b2[x+7];
            const float s7c = c0[x+5] + c0[x+6] + c0[x+7] + c1[x+5] + c1[x+6] + c1[x+7] + c2[x+5] + c2[x+6] + c2[x+7];
            
            // Pixel 8
            const float s8a = a0[x+6] + a0[x+7] + a0[x+8] + a1[x+6] + a1[x+7] + a1[x+8] + a2[x+6] + a2[x+7] + a2[x+8];
            const float s8b = b0[x+6] + b0[x+7] + b0[x+8] + b1[x+6] + b1[x+7] + b1[x+8] + b2[x+6] + b2[x+7] + b2[x+8];
            const float s8c = c0[x+6] + c0[x+7] + c0[x+8] + c1[x+6] + c1[x+7] + c1[x+8] + c2[x+6] + c2[x+7] + c2[x+8];
            
            // Accumulate box filter results
            total += (s1a * s1b) * inv81 + s1c * inv9;
            total += (s2a * s2b) * inv81 + s2c * inv9;
            total += (s3a * s3b) * inv81 + s3c * inv9;
            total += (s4a * s4b) * inv81 + s4c * inv9;
            total += (s5a * s5b) * inv81 + s5c * inv9;
            total += (s6a * s6b) * inv81 + s6c * inv9;
            total += (s7a * s7b) * inv81 + s7c * inv9;
            total += (s8a * s8b) * inv81 + s8c * inv9;
            
            // Sobel X gradients for 8 pixels
            for (int k = 0; k < 8; ++k) {
                size_t xk = x + k;
                float dx_d = -d0[xk-1] + d0[xk+1] - two*d1[xk-1] + two*d1[xk+1] - d2[xk-1] + d2[xk+1];
                float dx_e = -e0[xk-1] + e0[xk+1] - two*e1[xk-1] + two*e1[xk+1] - e2[xk-1] + e2[xk+1];
                float dx_f = -f0[xk-1] + f0[xk+1] - two*f1[xk-1] + two*f1[xk+1] - f2[xk-1] + f2[xk+1];
                total += dx_d * dx_e + dx_f;
                
                // Sobel Y gradients for 8 pixels
                float dy_g = -g0[xk-1] - two*g0[xk] - g0[xk+1] + g2[xk-1] + two*g2[xk] + g2[xk+1];
                float dy_h = -h0[xk-1] - two*h0[xk] - h0[xk+1] + h2[xk-1] + two*h2[xk] + h2[xk+1];
                float dy_i = -i0[xk-1] - two*i0[xk] - i0[xk+1] + i2[xk-1] + two*i2[xk] + i2[xk+1];
                total += dy_g * dy_h + dy_i;
            }
        }
        
        // Handle remaining pixels (0-7)
        for (; x + 1 < x_end; ++x) {
            const float sum_a = a0[x-1] + a0[x] + a0[x+1] + a1[x-1] + a1[x] + a1[x+1] + a2[x-1] + a2[x] + a2[x+1];
            const float sum_b = b0[x-1] + b0[x] + b0[x+1] + b1[x-1] + b1[x] + b1[x+1] + b2[x-1] + b2[x] + b2[x+1];
            const float sum_c = c0[x-1] + c0[x] + c0[x+1] + c1[x-1] + c1[x] + c1[x+1] + c2[x-1] + c2[x] + c2[x+1];
            total += sum_a * sum_b * inv81 + sum_c * inv9;
            
            const float dx_d = -d0[x-1] + d0[x+1] - two*d1[x-1] + two*d1[x+1] - d2[x-1] + d2[x+1];
            const float dx_e = -e0[x-1] + e0[x+1] - two*e1[x-1] + two*e1[x+1] - e2[x-1] + e2[x+1];
            const float dx_f = -f0[x-1] + f0[x+1] - two*f1[x-1] + two*f1[x+1] - f2[x-1] + f2[x+1];
            total += dx_d * dx_e + dx_f;
            
            const float dy_g = -g0[x-1] - two*g0[x] - g0[x+1] + g2[x-1] + two*g2[x] + g2[x+1];
            const float dy_h = -h0[x-1] - two*h0[x] - h0[x+1] + h2[x-1] + two*h2[x] + h2[x+1];
            const float dy_i = -i0[x-1] - two*i0[x] - i0[x+1] + i2[x-1] + two*i2[x] + i2[x+1];
            total += dy_g * dy_h + dy_i;
        }
    }

    out = total;
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
