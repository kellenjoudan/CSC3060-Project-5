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
    const std::size_t W = width;
    const std::size_t H = height;
    constexpr float inv9 = 1.0f / 9.0f;

    const auto& a = data.a;
    const auto& b = data.b;
    const auto& c = data.c;
    const auto& d = data.d;
    const auto& e = data.e;
    const auto& f = data.f;
    const auto& g = data.g;
    const auto& h = data.h;
    const auto& i_arr = data.i;

    double total = 0.0;

    for (std::size_t y = 1; y + 1 < H; ++y) {
        const std::size_t up = (y - 1) * W;
        const std::size_t mid = y * W;
        const std::size_t down = (y + 1) * W;

        for (std::size_t x = 1; x + 1 < W; ++x) {

            const std::size_t xm1 = x - 1;
            const std::size_t x0  = x;
            const std::size_t xp1 = x + 1;

            // 3x3 box filter (simple + compiler-friendly)
            float sum_a =
                a[up + xm1] + a[up + x0] + a[up + xp1] +
                a[mid + xm1] + a[mid + x0] + a[mid + xp1] +
                a[down + xm1] + a[down + x0] + a[down + xp1];

            float sum_b =
                b[up + xm1] + b[up + x0] + b[up + xp1] +
                b[mid + xm1] + b[mid + x0] + b[mid + xp1] +
                b[down + xm1] + b[down + x0] + b[down + xp1];

            float sum_c =
                c[up + xm1] + c[up + x0] + c[up + xp1] +
                c[mid + xm1] + c[mid + x0] + c[mid + xp1] +
                c[down + xm1] + c[down + x0] + c[down + xp1];

            float p1 = (sum_a * inv9) * (sum_b * inv9) + (sum_c * inv9);

            // Sobel X
            float dx_d =
                -d[up + xm1] + d[up + xp1]
                -2.0f * d[mid + xm1] + 2.0f * d[mid + xp1]
                -d[down + xm1] + d[down + xp1];

            float dx_e =
                -e[up + xm1] + e[up + xp1]
                -2.0f * e[mid + xm1] + 2.0f * e[mid + xp1]
                -e[down + xm1] + e[down + xp1];

            float dx_f =
                -f[up + xm1] + f[up + xp1]
                -2.0f * f[mid + xm1] + 2.0f * f[mid + xp1]
                -f[down + xm1] + f[down + xp1];

            float p2 = dx_d * dx_e + dx_f;

            // Sobel Y
            float gy_g =
                -g[up + xm1] - 2.0f * g[up + x0] - g[up + xp1]
                + g[down + xm1] + 2.0f * g[down + x0] + g[down + xp1];

            float gy_h =
                -h[up + xm1] - 2.0f * h[up + x0] - h[up + xp1]
                + h[down + xm1] + 2.0f * h[down + x0] + h[down + xp1];

            float gy_i =
                -i_arr[up + xm1] - 2.0f * i_arr[up + x0] - i_arr[up + xp1]
                + i_arr[down + xm1] + 2.0f * i_arr[down + x0] + i_arr[down + xp1];

            float p3 = gy_g * gy_h + gy_i;

            total += (p1 + p2 + p3);
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
