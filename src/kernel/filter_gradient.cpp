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

// Optimized Structure of Arrays (SoA) layout
struct optimized_data {
    // separate arrays for better SIMD
    std::vector<float> a, b, c, d, e, f, g, h, i;
    
    optimized_data(std::size_t width, std::size_t height) {
        size_t total_pixels = width * height;
        a.resize(total_pixels);
        b.resize(total_pixels);
        c.resize(total_pixels);
        d.resize(total_pixels);
        e.resize(total_pixels);
        f.resize(total_pixels);
        g.resize(total_pixels);
        h.resize(total_pixels);
        i.resize(total_pixels);
    }
};

// Conversion function
optimized_data convert_to_soa(const data_struct& naive, std::size_t width, std::size_t height) {
    optimized_data opt(width, height);
    size_t total_pixels = width * height;
    
    // Convert from AoS to SoA
    for (size_t idx = 0; idx < total_pixels; ++idx) {
        opt.a[idx] = naive.a[idx];
        opt.b[idx] = naive.b[idx];
        opt.c[idx] = naive.c[idx];
        opt.d[idx] = naive.d[idx];
        opt.e[idx] = naive.e[idx];
        opt.f[idx] = naive.f[idx];
        opt.g[idx] = naive.g[idx];
        opt.h[idx] = naive.h[idx];
        opt.i[idx] = naive.i[idx];
    }
    return opt;
}

void stu_filter_gradient(float& out, const data_struct& data,
                        std::size_t width, std::size_t height) {
    
    // Convert to optimized SoA layout
    optimized_data opt = convert_to_soa(data, width, height);
    
    const size_t W = width;
    const size_t H = height;
    constexpr float inv9 = 1.0f / 9.0f;
    constexpr float inv81 = inv9 * inv9;
    constexpr float two = 2.0f;
    
    float total = 0.0f;
    
    // Process in tiles for better cache locality
    constexpr size_t TILE_SIZE = 32;
    
    for (size_t y_tile = 1; y_tile + 1 < H; y_tile += TILE_SIZE) {
        size_t y_end = std::min(y_tile + TILE_SIZE, H - 1);
        
        for (size_t x_tile = 1; x_tile + 1 < W; x_tile += TILE_SIZE) {
            size_t x_end = std::min(x_tile + TILE_SIZE, W - 1);
            
            // Process tile
            for (size_t y = y_tile; y < y_end; ++y) {
                const size_t y_minus_1 = (y - 1) * W;
                const size_t y_curr = y * W;
                const size_t y_plus_1 = (y + 1) * W;
                
                // Get row pointers for all channels
                const float* a0 = opt.a.data() + y_minus_1;
                const float* a1 = opt.a.data() + y_curr;
                const float* a2 = opt.a.data() + y_plus_1;
                
                const float* b0 = opt.b.data() + y_minus_1;
                const float* b1 = opt.b.data() + y_curr;
                const float* b2 = opt.b.data() + y_plus_1;
                
                const float* c0 = opt.c.data() + y_minus_1;
                const float* c1 = opt.c.data() + y_curr;
                const float* c2 = opt.c.data() + y_plus_1;
                
                const float* d0 = opt.d.data() + y_minus_1;
                const float* d1 = opt.d.data() + y_curr;
                const float* d2 = opt.d.data() + y_plus_1;
                
                const float* e0 = opt.e.data() + y_minus_1;
                const float* e1 = opt.e.data() + y_curr;
                const float* e2 = opt.e.data() + y_plus_1;
                
                const float* f0 = opt.f.data() + y_minus_1;
                const float* f1 = opt.f.data() + y_curr;
                const float* f2 = opt.f.data() + y_plus_1;
                
                const float* g0 = opt.g.data() + y_minus_1;
                const float* g2 = opt.g.data() + y_plus_1;
                
                const float* h0 = opt.h.data() + y_minus_1;
                const float* h2 = opt.h.data() + y_plus_1;
                
                const float* i0 = opt.i.data() + y_minus_1;
                const float* i2 = opt.i.data() + y_plus_1;
                
                size_t x = std::max(x_tile, (size_t)1);
                const size_t x_end_local = std::min(x_end, W - 1);
                
                // Process 4 pixels at a time
                for (; x + 3 < x_end_local; x += 4) {
                    // PIXEL 1 (x)
                    float sum_a0 = a0[x-1] + a0[x] + a0[x+1] +
                                   a1[x-1] + a1[x] + a1[x+1] +
                                   a2[x-1] + a2[x] + a2[x+1];
                    
                    float sum_b0 = b0[x-1] + b0[x] + b0[x+1] +
                                   b1[x-1] + b1[x] + b1[x+1] +
                                   b2[x-1] + b2[x] + b2[x+1];
                    
                    float sum_c0 = c0[x-1] + c0[x] + c0[x+1] +
                                   c1[x-1] + c1[x] + c1[x+1] +
                                   c2[x-1] + c2[x] + c2[x+1];
                    
                    float p1_0 = sum_a0 * sum_b0 * inv81 + sum_c0 * inv9;
                    
                    float dx_d0 = -d0[x-1] + d0[x+1] - two*d1[x-1] + two*d1[x+1] - d2[x-1] + d2[x+1];
                    float dx_e0 = -e0[x-1] + e0[x+1] - two*e1[x-1] + two*e1[x+1] - e2[x-1] + e2[x+1];
                    float dx_f0 = -f0[x-1] + f0[x+1] - two*f1[x-1] + two*f1[x+1] - f2[x-1] + f2[x+1];
                    float p2_0 = dx_d0 * dx_e0 + dx_f0;
                    
                    float dy_g0 = -g0[x-1] - two*g0[x] - g0[x+1] + g2[x-1] + two*g2[x] + g2[x+1];
                    float dy_h0 = -h0[x-1] - two*h0[x] - h0[x+1] + h2[x-1] + two*h2[x] + h2[x+1];
                    float dy_i0 = -i0[x-1] - two*i0[x] - i0[x+1] + i2[x-1] + two*i2[x] + i2[x+1];
                    float p3_0 = dy_g0 * dy_h0 + dy_i0;
                    
                    // PIXEL 2 (x+1)
                    size_t x2 = x + 1;
                    float sum_a1 = a0[x2-1] + a0[x2] + a0[x2+1] +
                                   a1[x2-1] + a1[x2] + a1[x2+1] +
                                   a2[x2-1] + a2[x2] + a2[x2+1];
                    
                    float sum_b1 = b0[x2-1] + b0[x2] + b0[x2+1] +
                                   b1[x2-1] + b1[x2] + b1[x2+1] +
                                   b2[x2-1] + b2[x2] + b2[x2+1];
                    
                    float sum_c1 = c0[x2-1] + c0[x2] + c0[x2+1] +
                                   c1[x2-1] + c1[x2] + c1[x2+1] +
                                   c2[x2-1] + c2[x2] + c2[x2+1];
                    
                    float p1_1 = sum_a1 * sum_b1 * inv81 + sum_c1 * inv9;
                    
                    float dx_d1 = -d0[x2-1] + d0[x2+1] - two*d1[x2-1] + two*d1[x2+1] - d2[x2-1] + d2[x2+1];
                    float dx_e1 = -e0[x2-1] + e0[x2+1] - two*e1[x2-1] + two*e1[x2+1] - e2[x2-1] + e2[x2+1];
                    float dx_f1 = -f0[x2-1] + f0[x2+1] - two*f1[x2-1] + two*f1[x2+1] - f2[x2-1] + f2[x2+1];
                    float p2_1 = dx_d1 * dx_e1 + dx_f1;
                    
                    float dy_g1 = -g0[x2-1] - two*g0[x2] - g0[x2+1] + g2[x2-1] + two*g2[x2] + g2[x2+1];
                    float dy_h1 = -h0[x2-1] - two*h0[x2] - h0[x2+1] + h2[x2-1] + two*h2[x2] + h2[x2+1];
                    float dy_i1 = -i0[x2-1] - two*i0[x2] - i0[x2+1] + i2[x2-1] + two*i2[x2] + i2[x2+1];
                    float p3_1 = dy_g1 * dy_h1 + dy_i1;
                    
                    // PIXEL 3 (x+2)
                    size_t x3 = x + 2;
                    float sum_a2 = a0[x3-1] + a0[x3] + a0[x3+1] +
                                   a1[x3-1] + a1[x3] + a1[x3+1] +
                                   a2[x3-1] + a2[x3] + a2[x3+1];
                    
                    float sum_b2 = b0[x3-1] + b0[x3] + b0[x3+1] +
                                   b1[x3-1] + b1[x3] + b1[x3+1] +
                                   b2[x3-1] + b2[x3] + b2[x3+1];
                    
                    float sum_c2 = c0[x3-1] + c0[x3] + c0[x3+1] +
                                   c1[x3-1] + c1[x3] + c1[x3+1] +
                                   c2[x3-1] + c2[x3] + c2[x3+1];
                    
                    float p1_2 = sum_a2 * sum_b2 * inv81 + sum_c2 * inv9;
                    
                    float dx_d2 = -d0[x3-1] + d0[x3+1] - two*d1[x3-1] + two*d1[x3+1] - d2[x3-1] + d2[x3+1];
                    float dx_e2 = -e0[x3-1] + e0[x3+1] - two*e1[x3-1] + two*e1[x3+1] - e2[x3-1] + e2[x3+1];
                    float dx_f2 = -f0[x3-1] + f0[x3+1] - two*f1[x3-1] + two*f1[x3+1] - f2[x3-1] + f2[x3+1];
                    float p2_2 = dx_d2 * dx_e2 + dx_f2;
                    
                    float dy_g2 = -g0[x3-1] - two*g0[x3] - g0[x3+1] + g2[x3-1] + two*g2[x3] + g2[x3+1];
                    float dy_h2 = -h0[x3-1] - two*h0[x3] - h0[x3+1] + h2[x3-1] + two*h2[x3] + h2[x3+1];
                    float dy_i2 = -i0[x3-1] - two*i0[x3] - i0[x3+1] + i2[x3-1] + two*i2[x3] + i2[x3+1];
                    float p3_2 = dy_g2 * dy_h2 + dy_i2;
                    
                    // PIXEL 4 (x+3)
                    size_t x4 = x + 3;
                    float sum_a3 = a0[x4-1] + a0[x4] + a0[x4+1] +
                                   a1[x4-1] + a1[x4] + a1[x4+1] +
                                   a2[x4-1] + a2[x4] + a2[x4+1];
                    
                    float sum_b3 = b0[x4-1] + b0[x4] + b0[x4+1] +
                                   b1[x4-1] + b1[x4] + b1[x4+1] +
                                   b2[x4-1] + b2[x4] + b2[x4+1];
                    
                    float sum_c3 = c0[x4-1] + c0[x4] + c0[x4+1] +
                                   c1[x4-1] + c1[x4] + c1[x4+1] +
                                   c2[x4-1] + c2[x4] + c2[x4+1];
                    
                    float p1_3 = sum_a3 * sum_b3 * inv81 + sum_c3 * inv9;
                    
                    float dx_d3 = -d0[x4-1] + d0[x4+1] - two*d1[x4-1] + two*d1[x4+1] - d2[x4-1] + d2[x4+1];
                    float dx_e3 = -e0[x4-1] + e0[x4+1] - two*e1[x4-1] + two*e1[x4+1] - e2[x4-1] + e2[x4+1];
                    float dx_f3 = -f0[x4-1] + f0[x4+1] - two*f1[x4-1] + two*f1[x4+1] - f2[x4-1] + f2[x4+1];
                    float p2_3 = dx_d3 * dx_e3 + dx_f3;
                    
                    float dy_g3 = -g0[x4-1] - two*g0[x4] - g0[x4+1] + g2[x4-1] + two*g2[x4] + g2[x4+1];
                    float dy_h3 = -h0[x4-1] - two*h0[x4] - h0[x4+1] + h2[x4-1] + two*h2[x4] + h2[x4+1];
                    float dy_i3 = -i0[x4-1] - two*i0[x4] - i0[x4+1] + i2[x4-1] + two*i2[x4] + i2[x4+1];
                    float p3_3 = dy_g3 * dy_h3 + dy_i3;
                    
                    total += p1_0 + p2_0 + p3_0 + p1_1 + p2_1 + p3_1 + 
                             p1_2 + p2_2 + p3_2 + p1_3 + p2_3 + p3_3;
                }
                
                // Handle remaining pixels
                for (; x + 1 < x_end_local; ++x) {
                    float sum_a = a0[x-1] + a0[x] + a0[x+1] +
                                  a1[x-1] + a1[x] + a1[x+1] +
                                  a2[x-1] + a2[x] + a2[x+1];
                    
                    float sum_b = b0[x-1] + b0[x] + b0[x+1] +
                                  b1[x-1] + b1[x] + b1[x+1] +
                                  b2[x-1] + b2[x] + b2[x+1];
                    
                    float sum_c = c0[x-1] + c0[x] + c0[x+1] +
                                  c1[x-1] + c1[x] + c1[x+1] +
                                  c2[x-1] + c2[x] + c2[x+1];
                    
                    total += sum_a * sum_b * inv81 + sum_c * inv9;
                    
                    float dx_d = -d0[x-1] + d0[x+1] - two*d1[x-1] + two*d1[x+1] - d2[x-1] + d2[x+1];
                    float dx_e = -e0[x-1] + e0[x+1] - two*e1[x-1] + two*e1[x+1] - e2[x-1] + e2[x+1];
                    float dx_f = -f0[x-1] + f0[x+1] - two*f1[x-1] + two*f1[x+1] - f2[x-1] + f2[x+1];
                    total += dx_d * dx_e + dx_f;
                    
                    float dy_g = -g0[x-1] - two*g0[x] - g0[x+1] + g2[x-1] + two*g2[x] + g2[x+1];
                    float dy_h = -h0[x-1] - two*h0[x] - h0[x+1] + h2[x-1] + two*h2[x] + h2[x+1];
                    float dy_i = -i0[x-1] - two*i0[x] - i0[x+1] + i2[x-1] + two*i2[x] + i2[x+1];
                    total += dy_g * dy_h + dy_i;
                }
            }
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
