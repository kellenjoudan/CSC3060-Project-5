#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <print>
#include <vector>

#include "bench.h"
#include "relu.h"
#include "matmul.h"
#include "trace_replay.h"
#include "bitwise.h"
#include "blackscholes.h"
#include "filter_gradient.h"


int main() {
    std::uint32_t seed = 12345u;
    constexpr size_t relu_size = 1024000;
    relu_args relu_args_naive;
    initialize_relu(&relu_args_naive, relu_size, seed);
    std::println("\tReLU: vector length={}", relu_size);

    const std::size_t WIDTH = 1024;
    const std::size_t HEIGHT = 1024;
    filter_gradient_args filter_gradient_args_ref;
    initialize_filter_gradient(&filter_gradient_args_ref,
                               WIDTH,
                               HEIGHT,
                               seed);
    std::cout << "\tFilter Gradient: " << HEIGHT << " x " << WIDTH << '\n';


    std::vector<bench_t> benchmarks = {
                {"ReLU (Naive)",
                 naive_relu_wrapper,
                 naive_relu_wrapper,
                 relu_check,
                 &relu_args_naive,
                 &relu_args_naive,
                 BASELINE_RELU},

                {"Filter Gradient (Naive)",
                naive_filter_gradient_wrapper,
                naive_filter_gradient_wrapper,
                filter_gradient_check,
                &filter_gradient_args_ref,
                &filter_gradient_args_ref,
                BASELINE_FILTER_GRADIENT},

                {"Filter Gradient (Student)",
                stu_filter_gradient_wrapper,
                naive_filter_gradient_wrapper,
                filter_gradient_check,
                &filter_gradient_args_ref,
                &filter_gradient_args_ref,
                BASELINE_FILTER_GRADIENT},

    };
    std::cout << "\nRunning Benchmarks...\n";
    std::cout << "--------------------------------------------------------\n";
    std::cout << std::left << std::setw(25) << "Benchmark" << std::setw(12)
              << "Status" << std::right << std::setw(15) << "Nanoseconds"
              << "\n";
    std::cout << "--------------------------------------------------------\n";

    for (const auto &bench : benchmarks) {
        std::chrono::nanoseconds avg_time{0};
        const int k_best = 20;

        for (int i = 0; i < k_best; ++i) {
            flush_cache();
            const auto elapsed = measure_time([&] { bench.tfunc(bench.args); });

            avg_time += elapsed;
            debug_log("\tDEBUG: {}-th measurement: {} ns\n",
                      i,
                      static_cast<std::uint64_t>(elapsed.count()));
        }
        avg_time /= static_cast<uint64_t>(k_best);

        bool correct =
            bench.checkFunc(bench.args, bench.ref_args, bench.naiveFunc);

        std::cout << std::left << std::setw(25) << bench.description;
        if (!correct) {
            std::cout << "\033[1;31mFAILED\033[0m" << std::right
                      << std::setw(15) << "N/A" << "\n";
            std::cout
                << "  Error: Results do not match naive implementation!\n";
        } else {
            std::cout << "\033[1;32mPASSED\033[0m" << std::right
                      << std::setw(15) << avg_time.count() << " ns";
            if (avg_time.count() > bench.baseline_time.count() * 1.1) {
                std::cout << " (SLOW)";
            }
            std::cout << "\n";
        }
    }

    return 0;
}
