// Device numerical test for every anchor, both supported input sizes and row tails.
#include "rpp_box_decode.cuh"
#include <rpp_runtime.h>
#include <array>
#include <cstring>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {
void check(rtError_t result) { if (result != rtSuccess) throw std::runtime_error("RPP runtime operation failed"); }
struct Buffers {
    void* input = nullptr;
    void* output = nullptr;
    void* sram_input = nullptr;
    void* sram_output = nullptr;
    ~Buffers() {
        if (sram_input) rtFreeSram(sram_input);
        if (sram_output) rtFreeSram(sram_output);
        if (input) rtFree(input);
        if (output) rtFree(output);
    }
};
} // namespace

int main()
{
    try {
        check(rtSetDevice(0));
        for (int width : {640, 960}) {
            const int height = width;
            const int anchors = (width/8)*(height/8) + (width/16)*(height/16) + (width/32)*(height/32);
            const size_t input_bytes=size_t(64)*anchors*sizeof(float), bytes=size_t(4)*anchors*sizeof(float);
            Buffers buffers;
            check(rtMalloc(&buffers.input,input_bytes));
            check(rtMalloc(&buffers.output,bytes));
            check(rtMallocSram(&buffers.sram_input,input_bytes));
            check(rtMallocSram(&buffers.sram_output,bytes));
            for (int pass=0;pass<2;++pass) {
                std::vector<float> logits(64*anchors), expected(4*anchors), actual(4*anchors);
                for(int direction=0;direction<4;++direction) for(int i=0;i<anchors;++i) {
                    float sum=0, weighted=0;
                    const int peak=(i+direction*3+pass)%16;
                    for(int bin=0;bin<16;++bin) {
                        const float value=-.9f*std::abs(bin-peak);
                        logits[(direction*16+bin)*anchors+i]=value+(pass? -1000.0f:1000.0f);
                        // Match the FP32 subtraction used after storing logits.
                        const float e=std::exp(logits[(direction*16+bin)*anchors+i]-(pass?-1000.0f:1000.0f));
                        sum+=e; weighted+=e*bin;
                    }
                    expected[direction*anchors+i]=weighted/sum;
                }
                int start=0;
                for(int stride:{8,16,32}) {
                    for(int y=0;y<height/stride;++y) for(int x=0;x<width/stride;++x) {
                        const int i=start+y*(width/stride)+x;
                        const float l=expected[i],t=expected[anchors+i],r=expected[2*anchors+i],b=expected[3*anchors+i];
                        expected[i]=(x+.5f+(r-l)*.5f)*stride;
                        expected[anchors+i]=(y+.5f+(b-t)*.5f)*stride;
                        expected[2*anchors+i]=(l+r)*stride;
                        expected[3*anchors+i]=(t+b)*stride;
                    }
                    start+=(width/stride)*(height/stride);
                }
                check(rtMemcpy(buffers.input,logits.data(),input_bytes,rtMemcpyHostToDevice));
                check(rtMemcpy(buffers.sram_input,buffers.input,input_bytes,rtMemcpyDeviceToSram));
                launch_yolo11_dfl_decode(nullptr,static_cast<float*>(buffers.sram_input),
                    static_cast<float*>(buffers.sram_output),width,height,anchors);
                check(rtStreamSynchronize(nullptr));
                check(rtMemcpy(buffers.output,buffers.sram_output,bytes,rtMemcpySramToDevice));
                check(rtMemcpy(actual.data(),buffers.output,bytes,rtMemcpyDeviceToHost));
                float maximum=0;
                for(size_t i=0;i<actual.size();++i) {
                    if(!std::isfinite(actual[i])) throw std::runtime_error("Non-finite stable DFL output");
                    maximum=std::max(maximum,std::abs(actual[i]-expected[i]));
                }
                std::cout << "DFL FP32 max coordinate difference: " << maximum << '\n';
                if(maximum>.03f) throw std::runtime_error("Stable DFL reference mismatch");
            }
            std::cout << "PASS: " << width << "-square stable DFL, large positive/negative logits, all anchors\n";
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
