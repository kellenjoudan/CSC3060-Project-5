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

    double total = 0.0;

    // Get raw pointers to vector data
    const float* a_ptr = data.a.data();
    const float* b_ptr = data.b.data();
    const float* c_ptr = data.c.data();
    const float* d_ptr = data.d.data();
    const float* e_ptr = data.e.data();
    const float* f_ptr = data.f.data();
    const float* g_ptr = data.g.data();
    const float* h_ptr = data.h.data();
    const float* i_ptr = data.i.data();

    for (size_t y = 1; y + 1 < H; ++y) {
        const size_t y_minus_1 = (y - 1) * W;
        const size_t y_curr = y * W;
        const size_t y_plus_1 = (y + 1) * W;

        // Row pointers for each channel
        const float* a0 = a_ptr + y_minus_1;
        const float* a1 = a_ptr + y_curr;
        const float* a2 = a_ptr + y_plus_1;
        
        const float* b0 = b_ptr + y_minus_1;
        const float* b1 = b_ptr + y_curr;
        const float* b2 = b_ptr + y_plus_1;
        
        const float* c0 = c_ptr + y_minus_1;
        const float* c1 = c_ptr + y_curr;
        const float* c2 = c_ptr + y_plus_1;
        
        const float* d0 = d_ptr + y_minus_1;
        const float* d1 = d_ptr + y_curr;
        const float* d2 = d_ptr + y_plus_1;
        
        const float* e0 = e_ptr + y_minus_1;
        const float* e1 = e_ptr + y_curr;
        const float* e2 = e_ptr + y_plus_1;
        
        const float* f0 = f_ptr + y_minus_1;
        const float* f1 = f_ptr + y_curr;
        const float* f2 = f_ptr + y_plus_1;
        
        const float* g0 = g_ptr + y_minus_1;
        const float* g2 = g_ptr + y_plus_1;
        
        const float* h0 = h_ptr + y_minus_1;
        const float* h2 = h_ptr + y_plus_1;
        
        const float* i0 = i_ptr + y_minus_1;
        const float* i2 = i_ptr + y_plus_1;

        // Process 4 pixels at a time
        size_t x = 1;
        const size_t x_end = W - 1;
        
        for (; x + 3 < x_end; x += 4) {
            // Pixel 1
            const float a00_1 = a0[x-1], a01_1 = a0[x], a02_1 = a0[x+1];
            const float a10_1 = a1[x-1], a11_1 = a1[x], a12_1 = a1[x+1];
            const float a20_1 = a2[x-1], a21_1 = a2[x], a22_1 = a2[x+1];
            const float sum_a0 = a00_1 + a01_1 + a02_1 + a10_1 + a11_1 + a12_1 + a20_1 + a21_1 + a22_1;
            
            const float b00_1 = b0[x-1], b01_1 = b0[x], b02_1 = b0[x+1];
            const float b10_1 = b1[x-1], b11_1 = b1[x], b12_1 = b1[x+1];
            const float b20_1 = b2[x-1], b21_1 = b2[x], b22_1 = b2[x+1];
            const float sum_b0 = b00_1 + b01_1 + b02_1 + b10_1 + b11_1 + b12_1 + b20_1 + b21_1 + b22_1;
            
            const float c00_1 = c0[x-1], c01_1 = c0[x], c02_1 = c0[x+1];
            const float c10_1 = c1[x-1], c11_1 = c1[x], c12_1 = c1[x+1];
            const float c20_1 = c2[x-1], c21_1 = c2[x], c22_1 = c2[x+1];
            const float sum_c0 = c00_1 + c01_1 + c02_1 + c10_1 + c11_1 + c12_1 + c20_1 + c21_1 + c22_1;
            
            const float p1_0 = sum_a0 * sum_b0 * inv81 + sum_c0 * inv9;
            
            const float dx_d0 = -d0[x-1] + d0[x+1] - 2.0f*d1[x-1] + 2.0f*d1[x+1] - d2[x-1] + d2[x+1];
            const float dx_e0 = -e0[x-1] + e0[x+1] - 2.0f*e1[x-1] + 2.0f*e1[x+1] - e2[x-1] + e2[x+1];
            const float dx_f0 = -f0[x-1] + f0[x+1] - 2.0f*f1[x-1] + 2.0f*f1[x+1] - f2[x-1] + f2[x+1];
            const float p2_0 = dx_d0 * dx_e0 + dx_f0;
            
            const float dy_g0 = -g0[x-1] - 2.0f*g0[x] - g0[x+1] + g2[x-1] + 2.0f*g2[x] + g2[x+1];
            const float dy_h0 = -h0[x-1] - 2.0f*h0[x] - h0[x+1] + h2[x-1] + 2.0f*h2[x] + h2[x+1];
            const float dy_i0 = -i0[x-1] - 2.0f*i0[x] - i0[x+1] + i2[x-1] + 2.0f*i2[x] + i2[x+1];
            const float p3_0 = dy_g0 * dy_h0 + dy_i0;
            
            // Pixel 2
            const size_t x2 = x + 1;
            const float a00_2 = a0[x2-1], a01_2 = a0[x2], a02_2 = a0[x2+1];
            const float a10_2 = a1[x2-1], a11_2 = a1[x2], a12_2 = a1[x2+1];
            const float a20_2 = a2[x2-1], a21_2 = a2[x2], a22_2 = a2[x2+1];
            const float sum_a1 = a00_2 + a01_2 + a02_2 + a10_2 + a11_2 + a12_2 + a20_2 + a21_2 + a22_2;
            
            const float b00_2 = b0[x2-1], b01_2 = b0[x2], b02_2 = b0[x2+1];
            const float b10_2 = b1[x2-1], b11_2 = b1[x2], b12_2 = b1[x2+1];
            const float b20_2 = b2[x2-1], b21_2 = b2[x2], b22_2 = b2[x2+1];
            const float sum_b1 = b00_2 + b01_2 + b02_2 + b10_2 + b11_2 + b12_2 + b20_2 + b21_2 + b22_2;
            
            const float c00_2 = c0[x2-1], c01_2 = c0[x2], c02_2 = c0[x2+1];
            const float c10_2 = c1[x2-1], c11_2 = c1[x2], c12_2 = c1[x2+1];
            const float c20_2 = c2[x2-1], c21_2 = c2[x2], c22_2 = c2[x2+1];
            const float sum_c1 = c00_2 + c01_2 + c02_2 + c10_2 + c11_2 + c12_2 + c20_2 + c21_2 + c22_2;
            
            const float p1_1 = sum_a1 * sum_b1 * inv81 + sum_c1 * inv9;
            
            const float dx_d1 = -d0[x2-1] + d0[x2+1] - 2.0f*d1[x2-1] + 2.0f*d1[x2+1] - d2[x2-1] + d2[x2+1];
            const float dx_e1 = -e0[x2-1] + e0[x2+1] - 2.0f*e1[x2-1] + 2.0f*e1[x2+1] - e2[x2-1] + e2[x2+1];
            const float dx_f1 = -f0[x2-1] + f0[x2+1] - 2.0f*f1[x2-1] + 2.0f*f1[x2+1] - f2[x2-1] + f2[x2+1];
            const float p2_1 = dx_d1 * dx_e1 + dx_f1;
            
            const float dy_g1 = -g0[x2-1] - 2.0f*g0[x2] - g0[x2+1] + g2[x2-1] + 2.0f*g2[x2] + g2[x2+1];
            const float dy_h1 = -h0[x2-1] - 2.0f*h0[x2] - h0[x2+1] + h2[x2-1] + 2.0f*h2[x2] + h2[x2+1];
            const float dy_i1 = -i0[x2-1] - 2.0f*i0[x2] - i0[x2+1] + i2[x2-1] + 2.0f*i2[x2] + i2[x2+1];
            const float p3_1 = dy_g1 * dy_h1 + dy_i1;
            
            // Pixel 3
            const size_t x3 = x + 2;
            const float sum_a2 = (a0[x3-1] + a0[x3] + a0[x3+1]) + (a1[x3-1] + a1[x3] + a1[x3+1]) + (a2[x3-1] + a2[x3] + a2[x3+1]);
            const float sum_b2 = (b0[x3-1] + b0[x3] + b0[x3+1]) + (b1[x3-1] + b1[x3] + b1[x3+1]) + (b2[x3-1] + b2[x3] + b2[x3+1]);
            const float sum_c2 = (c0[x3-1] + c0[x3] + c0[x3+1]) + (c1[x3-1] + c1[x3] + c1[x3+1]) + (c2[x3-1] + c2[x3] + c2[x3+1]);
            const float p1_2 = sum_a2 * sum_b2 * inv81 + sum_c2 * inv9;
            
            const float dx_d2 = -d0[x3-1] + d0[x3+1] - 2.0f*d1[x3-1] + 2.0f*d1[x3+1] - d2[x3-1] + d2[x3+1];
            const float dx_e2 = -e0[x3-1] + e0[x3+1] - 2.0f*e1[x3-1] + 2.0f*e1[x3+1] - e2[x3-1] + e2[x3+1];
            const float dx_f2 = -f0[x3-1] + f0[x3+1] - 2.0f*f1[x3-1] + 2.0f*f1[x3+1] - f2[x3-1] + f2[x3+1];
            const float p2_2 = dx_d2 * dx_e2 + dx_f2;
            
            const float dy_g2 = -g0[x3-1] - 2.0f*g0[x3] - g0[x3+1] + g2[x3-1] + 2.0f*g2[x3] + g2[x3+1];
            const float dy_h2 = -h0[x3-1] - 2.0f*h0[x3] - h0[x3+1] + h2[x3-1] + 2.0f*h2[x3] + h2[x3+1];
            const float dy_i2 = -i0[x3-1] - 2.0f*i0[x3] - i0[x3+1] + i2[x3-1] + 2.0f*i2[x3] + i2[x3+1];
            const float p3_2 = dy_g2 * dy_h2 + dy_i2;
            
            // Pixel 4
            const size_t x4 = x + 3;
            const float sum_a3 = (a0[x4-1] + a0[x4] + a0[x4+1]) + (a1[x4-1] + a1[x4] + a1[x4+1]) + (a2[x4-1] + a2[x4] + a2[x4+1]);
            const float sum_b3 = (b0[x4-1] + b0[x4] + b0[x4+1]) + (b1[x4-1] + b1[x4] + b1[x4+1]) + (b2[x4-1] + b2[x4] + b2[x4+1]);
            const float sum_c3 = (c0[x4-1] + c0[x4] + c0[x4+1]) + (c1[x4-1] + c1[x4] + c1[x4+1]) + (c2[x4-1] + c2[x4] + c2[x4+1]);
            const float p1_3 = sum_a3 * sum_b3 * inv81 + sum_c3 * inv9;
            
            const float dx_d3 = -d0[x4-1] + d0[x4+1] - 2.0f*d1[x4-1] + 2.0f*d1[x4+1] - d2[x4-1] + d2[x4+1];
            const float dx_e3 = -e0[x4-1] + e0[x4+1] - 2.0f*e1[x4-1] + 2.0f*e1[x4+1] - e2[x4-1] + e2[x4+1];
            const float dx_f3 = -f0[x4-1] + f0[x4+1] - 2.0f*f1[x4-1] + 2.0f*f1[x4+1] - f2[x4-1] + f2[x4+1];
            const float p2_3 = dx_d3 * dx_e3 + dx_f3;
            
            const float dy_g3 = -g0[x4-1] - 2.0f*g0[x4] - g0[x4+1] + g2[x4-1] + 2.0f*g2[x4] + g2[x4+1];
            const float dy_h3 = -h0[x4-1] - 2.0f*h0[x4] - h0[x4+1] + h2[x4-1] + 2.0f*h2[x4] + h2[x4+1];
            const float dy_i3 = -i0[x4-1] - 2.0f*i0[x4] - i0[x4+1] + i2[x4-1] + 2.0f*i2[x4] + i2[x4+1];
            const float p3_3 = dy_g3 * dy_h3 + dy_i3;
            
            total += p1_0 + p2_0 + p3_0 + p1_1 + p2_1 + p3_1 + p1_2 + p2_2 + p3_2 + p1_3 + p2_3 + p3_3;
        }
        
        // Handle remaining pixels
        for (; x + 1 < x_end; ++x) {
            const float sum_a = (a0[x-1] + a0[x] + a0[x+1]) + (a1[x-1] + a1[x] + a1[x+1]) + (a2[x-1] + a2[x] + a2[x+1]);
            const float sum_b = (b0[x-1] + b0[x] + b0[x+1]) + (b1[x-1] + b1[x] + b1[x+1]) + (b2[x-1] + b2[x] + b2[x+1]);
            const float sum_c = (c0[x-1] + c0[x] + c0[x+1]) + (c1[x-1] + c1[x] + c1[x+1]) + (c2[x-1] + c2[x] + c2[x+1]);
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
