// Build-time helper. The SDK assembler is used only on the host during build.
#include <string>
#include <vector>

extern bool asmRtExec(std::string, std::vector<std::string>, std::string);

int main(int argc, char** argv)
{
    if (argc != 2) return 2;
    return asmRtExec("asm/kernel_cfg.json", {argv[1]}, "./") ? 0 : 1;
}
