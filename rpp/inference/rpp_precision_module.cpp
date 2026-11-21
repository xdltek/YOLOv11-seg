#include "rpp_precision_module.h"

#include <rpp_drv_api.h>
#include <dlfcn.h>
#include <elf.h>

#include <cstring>
#include <fstream>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace {
thread_local bool precision_enabled = false;

struct PrecisionKernel {
    const char* name;
    const unsigned char* original;
    size_t original_size;
    const unsigned char* replacement;
    size_t replacement_size;
    const unsigned char* metadata;
    size_t metadata_size;
    const char* source_path;
    const char* source;
};
#include "rpp_precision_kernels.inc"

bool within(size_t offset, size_t bytes, size_t size)
{
    return offset <= size && bytes <= size - offset;
}

template<class T> T read(const std::vector<char>& image, size_t offset)
{
    if (!within(offset, sizeof(T), image.size())) throw std::runtime_error("Truncated ELF record");
    T value;
    std::memcpy(&value, image.data() + offset, sizeof(value));
    return value;
}
} // namespace

namespace yolo11 {
ScopedConvolutionPrecision::ScopedConvolutionPrecision(bool enabled) : previous_(precision_enabled)
{
    precision_enabled = enabled;
}
ScopedConvolutionPrecision::~ScopedConvolutionPrecision() { precision_enabled = previous_; }

size_t correctConvolutionModule(std::vector<char>& image, const std::string& assembly_root)
{
    const auto header = read<Elf64_Ehdr>(image, 0);
    if (std::memcmp(header.e_ident, ELFMAG, SELFMAG) || header.e_ident[EI_CLASS] != ELFCLASS64 ||
        header.e_ident[EI_DATA] != ELFDATA2LSB || header.e_type != ET_REL ||
        header.e_shentsize != sizeof(Elf64_Shdr) || !header.e_shnum ||
        header.e_shstrndx >= header.e_shnum ||
        !within(header.e_shoff, size_t(header.e_shnum) * sizeof(Elf64_Shdr), image.size()))
        throw std::runtime_error("Unsupported RPP module ELF format");
    std::vector<Elf64_Shdr> sections;
    for (unsigned i = 0; i < header.e_shnum; ++i) {
        auto section = read<Elf64_Shdr>(image, header.e_shoff + i * sizeof(Elf64_Shdr));
        if (section.sh_type != SHT_NOBITS && !within(section.sh_offset, section.sh_size, image.size()))
            throw std::runtime_error("Invalid ELF section range");
        sections.push_back(section);
    }
    const auto strings = sections[header.e_shstrndx];
    if (strings.sh_type != SHT_STRTAB) throw std::runtime_error("Missing ELF section names");
    std::unordered_map<std::string, size_t> names;
    for (size_t i = 0; i < sections.size(); ++i) {
        const auto offset = sections[i].sh_name;
        if (offset >= strings.sh_size) throw std::runtime_error("Invalid ELF section name");
        const char* name = image.data() + strings.sh_offset + offset;
        if (!std::memchr(name, 0, strings.sh_size - offset)) throw std::runtime_error("Unterminated ELF section name");
        if (!names.emplace(name, i).second) throw std::runtime_error("Duplicate ELF section name");
    }
    std::vector<std::pair<size_t, const PrecisionKernel*>> patches;
    for (const auto& kernel : kPrecisionKernels) {
        auto code = names.find(std::string(".text.") + kernel.name);
        if (code == names.end()) continue;
        auto info = names.find(std::string(".rpp.info.") + kernel.name);
        const auto& text = sections[code->second];
        if (info == names.end() || text.sh_type != SHT_PROGBITS || text.sh_size != kernel.original_size)
            throw std::runtime_error(std::string("SDK convolution code changed: ") + kernel.name);
        const auto& metadata = sections[info->second];
        if (metadata.sh_type != SHT_PROGBITS || metadata.sh_size != kernel.metadata_size ||
            std::memcmp(image.data() + metadata.sh_offset, kernel.metadata, kernel.metadata_size))
            throw std::runtime_error(std::string("SDK convolution ABI changed: ") + kernel.name);
        // OpenRT copies the SDK assembly into this process's temporary module
        // directory. Compare the actual source used, since unused instruction
        // bits emitted by the assembler vary between independent builds.
        std::ifstream source_file(std::filesystem::path(assembly_root) / kernel.source_path,
                                  std::ios::binary);
        const std::string source((std::istreambuf_iterator<char>(source_file)), {});
        if (!source_file || source != kernel.source)
            throw std::runtime_error(std::string("SDK convolution source changed: ") + kernel.name);
        patches.emplace_back(code->second, &kernel);
    }
    // Complete every compatibility check before changing anything. Append each
    // replacement section, preserving all original metadata and other kernels.
    for (const auto& patch : patches) {
        auto& section = sections[patch.first];
        const auto& kernel = *patch.second;
        const auto alignment = section.sh_addralign ? section.sh_addralign : 1;
        if (alignment > 4096 || (alignment & (alignment - 1)))
            throw std::runtime_error("Invalid ELF section alignment");
        image.resize((image.size() + alignment - 1) & ~(alignment - 1), 0);
        section.sh_offset = image.size();
        section.sh_size = kernel.replacement_size;
        image.insert(image.end(), kernel.replacement, kernel.replacement + kernel.replacement_size);
    }
    for (const auto& patch : patches)
        std::memcpy(image.data() + header.e_shoff + patch.first * sizeof(Elf64_Shdr),
                    &sections[patch.first], sizeof(Elf64_Shdr));
    return patches.size();
}
} // namespace yolo11

// OpenRT builds a process-local RPP ELF module at first execution. Keep OpenRT
// scheduling/bindings unchanged and correct only verified convolution code in
// that module. This symbol interposes the driver's public module loader inside
// this application; it neither installs nor overwrites any SDK file.
extern "C" RPPresult rppModuleLoad(RPPmodule* module, const char* path)
{
    using Loader = RPPresult (*)(RPPmodule*, const char*);
    static const auto original = reinterpret_cast<Loader>(dlsym(RTLD_NEXT, "rppModuleLoad"));
    if (!original) return RPP_ERROR_NOT_INITIALIZED;
    if (!precision_enabled || !path || std::string(path).find("/rpp_bin.o") == std::string::npos)
        return original(module, path);
    try {
        std::ifstream input(path, std::ios::binary | std::ios::ate);
        if (!input || input.tellg() <= 0 || input.tellg() > std::streampos(16 * 1024 * 1024))
            throw std::runtime_error("Cannot read OpenRT kernel module");
        std::vector<char> image(static_cast<size_t>(input.tellg()));
        input.seekg(0);
        if (!input.read(image.data(), image.size())) throw std::runtime_error("Incomplete kernel module");
        const auto assembly_root = std::filesystem::path(path).parent_path() / "asm";
        const auto count = yolo11::correctConvolutionModule(image, assembly_root.string());
        if (!count) return original(module, path);
        const auto result = rppModuleLoadData(module, image.data());
        if (result == RPP_SUCCESS)
            std::cout << "[ENGINE  ] FP32 bias accumulation and nearest BF16 rounding: "
                      << count << " convolution kernels" << std::endl;
        return result;
    } catch (const std::exception& error) {
        std::cerr << "[ENGINE  ] Convolution precision initialization failed: " << error.what() << std::endl;
        return RPP_ERROR_INVALID_SOURCE;
    }
}
