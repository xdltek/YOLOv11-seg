// Verify device prefix/scatter across row tails, empty input and >512 positives.
#include "rpp_postprocess_kernels.cuh"
#include <rpp_runtime.h>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>

int main()
{
    void *device=nullptr,*input=nullptr,*output=nullptr,*ping=nullptr,*pong=nullptr,*map=nullptr;
    auto check=[](rtError_t status) { if(status!=rtSuccess) throw std::runtime_error("RPP operation failed"); };
    auto release=[&]() { for(void* p:{input,output,ping,pong,map}) if(p) rtFreeSram(p); if(device) rtFree(device); };
    try {
        check(rtSetDevice(0));
        constexpr int anchors=18900,classes=80,channels=84;
        const size_t bytes=size_t(channels)*anchors*sizeof(float),index_bytes=size_t(anchors)*sizeof(int);
        check(rtMalloc(&device,bytes));
        check(rtMallocSram(&input,bytes));check(rtMallocSram(&output,bytes));
        check(rtMallocSram(&ping,index_bytes));check(rtMallocSram(&pong,index_bytes));check(rtMallocSram(&map,index_bytes));
        for(int pass=0;pass<3;++pass) {
            std::vector<float> source(channels*anchors),expected(channels*anchors),actual(channels*anchors);
            std::vector<int> expected_map(anchors,-1),actual_map(anchors);
            int count=0;
            for(int i=0;i<anchors;++i) {
                source[i]=float(i%1000);source[anchors+i]=100;source[2*anchors+i]=20;source[3*anchors+i]=30;
                const bool valid=pass==1 ? (i%139==0 || i==anchors-1) : pass==2 && i%5==0;
                if(!valid) continue;
                source[(4+i%classes)*anchors+i]=(i%2)? .25f:.9f;
                expected_map[count]=i;
                for(int c=0;c<channels;++c) expected[c*anchors+count]=source[c*anchors+i];
                ++count;
            }
            check(rtMemcpy(device,source.data(),bytes,rtMemcpyHostToDevice));
            check(rtMemcpy(input,device,bytes,rtMemcpyDeviceToSram));
            launch_yolov11_seg_compact_candidates(nullptr,static_cast<float*>(input),static_cast<float*>(output),
                static_cast<int*>(ping),static_cast<int*>(pong),static_cast<int*>(map),anchors,classes,.25f);
            check(rtStreamSynchronize(nullptr));
            check(rtMemcpy(device,output,bytes,rtMemcpySramToDevice));
            check(rtMemcpy(actual.data(),device,bytes,rtMemcpyDeviceToHost));
            check(rtMemcpy(device,map,index_bytes,rtMemcpySramToDevice));
            check(rtMemcpy(actual_map.data(),device,index_bytes,rtMemcpyDeviceToHost));
            if(actual_map!=expected_map || std::memcmp(actual.data(),expected.data(),bytes))
                throw std::runtime_error("Device compaction differs from stable CPU reference");
            std::cout << "PASS: " << count << " candidates, all planes and full original-anchor map exact\n";
        }
        release();return 0;
    } catch(const std::exception& e) { std::cerr<<e.what()<<'\n';release();return 1; }
}
