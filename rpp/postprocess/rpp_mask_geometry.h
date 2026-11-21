#ifndef YOLO11_RPP_MASK_GEOMETRY_H
#define YOLO11_RPP_MASK_GEOMETRY_H

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace yolo11 {
// Two consecutive model pixels cover at most three prototype pixels when
// prototype <= model. Merge repeated taps of the two bilinear resizes once.
constexpr int kMaskResizeTaps = 3;

inline void buildMaskResizeAxis(int source, int resized, int model, int prototype,
                                int pad, std::vector<int>& indices,
                                std::vector<float>& weights) {
    if (source <= 0 || resized <= 0 || prototype <= 0 || model < prototype ||
        pad < 0 || pad + resized > model)
        throw std::runtime_error("Invalid prototype/model/source resize geometry");
    indices.resize(size_t(source) * kMaskResizeTaps);
    weights.resize(size_t(source) * kMaskResizeTaps);
    for (int s = 0; s < source; ++s) {
        const double cropped = std::clamp((s + 0.5) * resized / source - 0.5,
                                          0.0, double(resized - 1));
        const int lower = int(std::floor(cropped));
        const double alpha = cropped - lower;
        std::array<int, kMaskResizeTaps> taps{};
        std::array<float, kMaskResizeTaps> merged{};
        int count = 0;
        for (int k = 0; k < 2; ++k) {
            const int m = std::min(lower + k, resized - 1) + pad;
            const double p = std::clamp((m + 0.5) * prototype / model - 0.5,
                                        0.0, double(prototype - 1));
            const int p0 = int(std::floor(p));
            const double beta = p - p0;
            for (int j = 0; j < 2; ++j) {
                const int index = std::min(p0 + j, prototype - 1);
                const float weight = float((k ? alpha : 1 - alpha) *
                                            (j ? beta : 1 - beta));
                int slot = 0;
                while (slot < count && taps[slot] != index) ++slot;
                if (slot == count) {
                    if (count == kMaskResizeTaps)
                        throw std::runtime_error("Resize exceeds three prototype taps");
                    taps[count++] = index;
                }
                merged[slot] += weight;
            }
        }
        for (int j = 0; j < kMaskResizeTaps; ++j) {
            indices[size_t(s) * kMaskResizeTaps + j] = taps[std::min(j, count - 1)];
            weights[size_t(s) * kMaskResizeTaps + j] = merged[j];
        }
    }
}
} // namespace yolo11
#endif
