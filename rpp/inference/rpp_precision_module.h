#ifndef YOLO11_RPP_PRECISION_MODULE_H
#define YOLO11_RPP_PRECISION_MODULE_H

#include <cstddef>
#include <string>
#include <vector>

namespace yolo11 {

// Restrict the module correction to synchronous inference initialization. Other
// SDK users in the same process, including the POST engine, retain their kernels.
class ScopedConvolutionPrecision {
public:
    explicit ScopedConvolutionPrecision(bool enabled = true);
    ~ScopedConvolutionPrecision();
    ScopedConvolutionPrecision(const ScopedConvolutionPrecision&) = delete;
    ScopedConvolutionPrecision& operator=(const ScopedConvolutionPrecision&) = delete;
private:
    bool previous_;
};

// Host-only ELF preparation. Throws on malformed input, changed source, code size
// or parameter ABI. Exposed to the diagnostic test without executing the device.
size_t correctConvolutionModule(std::vector<char>& image, const std::string& assembly_root);

} // namespace yolo11
#endif
