#include "rpp_model_support.h"
#include <iostream>

int main()
{
    // Positive release profiles, including both m/l stem scales.
    for (int stem : {16, 32, 64, 64, 96})
        if (!yolo11::supportsModelProfile(640, 640, stem)) return 1;
    for (int stem : {16, 32})
        if (!yolo11::supportsModelProfile(960, 960, stem)) return 1;
    // Near-boundary scales and dimensions must not enter an engine build.
    for (int stem : {0, 48, 64, 96, 128})
        if (yolo11::supportsModelProfile(960, 960, stem)) return 1;
    if (yolo11::supportsModelProfile(640, 640, 48) ||
        yolo11::supportsModelProfile(960, 640, 16) ||
        yolo11::supportsModelProfile(1280, 1024, 16)) return 1;
    std::cout << "PASS: seven release profiles and boundary rejection\n";
    return 0;
}
