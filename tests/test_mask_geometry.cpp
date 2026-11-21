// Compare merged interpolation taps with two independent bilinear resizes.
#include "rpp_mask_geometry.h"

#include <cmath>
#include <iostream>
#include <stdexcept>

int main() {
    try {
        double maximum = 0;
        for (int model : {640, 960}) for (int source : {1, 8, 810, 1080, 1280, 1920})
            for (int resized : {model / 2, model * 3 / 4, model}) {
                const int prototype = model / 4, pad = (model - resized) / 2;
                std::vector<int> indices;
                std::vector<float> weights;
                yolo11::buildMaskResizeAxis(source, resized, model, prototype, pad, indices, weights);
                // Nonlinear samples expose accidental index or weight substitutions.
                auto sample = [](int x) { return std::sin(x * 0.71) * 0.5 + 0.5; };
                auto model_sample = [&](int m) {
                    const double p = std::clamp((m + 0.5) * prototype / model - 0.5,
                                                0.0, double(prototype - 1));
                    const int p0 = int(std::floor(p));
                    return sample(p0) * (1 - (p - p0)) +
                           sample(std::min(p0 + 1, prototype - 1)) * (p - p0);
                };
                for (int s = 0; s < source; ++s) {
                    const double p = std::clamp((s + 0.5) * resized / source - 0.5,
                                                0.0, double(resized - 1));
                    const int p0 = int(std::floor(p));
                    const double reference = model_sample(p0 + pad) * (1 - (p - p0)) +
                        model_sample(std::min(p0 + 1, resized - 1) + pad) * (p - p0);
                    double actual = 0, sum = 0;
                    for (int j = 0; j < yolo11::kMaskResizeTaps; ++j) {
                        const auto offset = size_t(s) * yolo11::kMaskResizeTaps + j;
                        if (indices[offset] < 0 || indices[offset] >= prototype || weights[offset] < 0)
                            throw std::runtime_error("Invalid merged resize tap");
                        actual += sample(indices[offset]) * weights[offset];
                        sum += weights[offset];
                    }
                    maximum = std::max(maximum, std::abs(actual - reference));
                    if (std::abs(sum - 1) > 2e-7 || std::abs(actual - reference) > 2e-7)
                        throw std::runtime_error("Merged interpolation differs from independent reference");
                }
            }
        std::cout << "PASS: merged resize weights, edge clamping and padding; max error=" << maximum << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
