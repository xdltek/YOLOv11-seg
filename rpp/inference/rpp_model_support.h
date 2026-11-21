#ifndef YOLO11_RPP_MODEL_SUPPORT_H
#define YOLO11_RPP_MODEL_SUPPORT_H

namespace yolo11 {
// The static YOLO11 export's first convolution identifies its width scale:
// n=16, s=32, m/l=64, x=96. Inspect the graph rather than the model filename.
constexpr bool supportsModelProfile(int width, int height, int stem_channels)
{
    return (width == 640 && height == 640 &&
            (stem_channels == 16 || stem_channels == 32 || stem_channels == 64 || stem_channels == 96)) ||
           (width == 960 && height == 960 && (stem_channels == 16 || stem_channels == 32));
}
} // namespace yolo11
#endif
