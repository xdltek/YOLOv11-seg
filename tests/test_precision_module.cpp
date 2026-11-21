// Host-only contract checks: no RPP context or inference is created.
#include "rpp_precision_module.h"
#include <elf.h>
#include <unistd.h>

#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
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

void require(bool value, const char* message)
{
    if (!value) throw std::runtime_error(message);
}

std::vector<char> fixture()
{
    const auto& kernel = kPrecisionKernels[0];
    Elf64_Ehdr header{};
    std::memcpy(header.e_ident, ELFMAG, SELFMAG);
    header.e_ident[EI_CLASS] = ELFCLASS64;
    header.e_ident[EI_DATA] = ELFDATA2LSB;
    header.e_type = ET_REL;
    header.e_shentsize = sizeof(Elf64_Shdr);
    header.e_shoff = sizeof(header);
    header.e_shnum = 5;
    header.e_shstrndx = 1;
    std::vector<char> image(sizeof(header) + 5 * sizeof(Elf64_Shdr), 0);
    std::string strings(1, '\0');
    std::vector<Elf64_Shdr> sections(5);
    auto name = [&](int index, const std::string& value) {
        sections[index].sh_name = strings.size();
        strings += value;
        strings.push_back(0);
    };
    name(1, ".shstrtab");
    name(2, std::string(".text.") + kernel.name);
    name(3, std::string(".rpp.info.") + kernel.name);
    name(4, ".text.unrelated");
    auto append = [&](int index, const void* data, size_t size) {
        sections[index].sh_offset = image.size();
        sections[index].sh_size = size;
        sections[index].sh_type = SHT_PROGBITS;
        sections[index].sh_addralign = 1;
        const auto* bytes = static_cast<const char*>(data);
        image.insert(image.end(), bytes, bytes + size);
    };
    append(1, strings.data(), strings.size());
    sections[1].sh_type = SHT_STRTAB;
    append(2, kernel.original, kernel.original_size);
    append(3, kernel.metadata, kernel.metadata_size);
    append(4, "UNCHANGED", 9);
    std::memcpy(image.data(), &header, sizeof(header));
    std::memcpy(image.data() + header.e_shoff, sections.data(), sections.size() * sizeof(Elf64_Shdr));
    return image;
}

Elf64_Shdr section(const std::vector<char>& image, int index)
{
    Elf64_Shdr result;
    std::memcpy(&result, image.data() + sizeof(Elf64_Ehdr) + index * sizeof(result), sizeof(result));
    return result;
}

void mustReject(std::vector<char> image, const std::string& source)
{
    const auto before = image;
    bool rejected = false;
    try { yolo11::correctConvolutionModule(image, source); }
    catch (const std::runtime_error&) { rejected = true; }
    require(rejected, "Invalid module was accepted");
    require(before == image, "Rejected module was modified");
}
} // namespace

int main()
{
    const auto root = std::filesystem::temp_directory_path() /
                      ("yolo11_precision_test_" + std::to_string(getpid()));
    try {
        const auto& kernel = kPrecisionKernels[0];
        const auto source = root / kernel.source_path;
        std::filesystem::create_directories(source.parent_path());
        { std::ofstream file(source, std::ios::binary); file << kernel.source; }
        auto image = fixture();
        const auto original = image;
        require(yolo11::correctConvolutionModule(image, root.string()) == 1, "Expected one correction");
        const auto code = section(image, 2);
        require(code.sh_size == kernel.replacement_size &&
                !std::memcmp(image.data() + code.sh_offset, kernel.replacement, code.sh_size),
                "Corrected code does not match the assembled kernel");
        const auto metadata = section(image, 3);
        require(!std::memcmp(image.data() + metadata.sh_offset, kernel.metadata, metadata.sh_size),
                "Parameter ABI changed");
        const auto other = section(image, 4);
        require(!std::memcmp(image.data() + other.sh_offset, "UNCHANGED", 9), "Unrelated kernel changed");
        auto invalid = original;
        invalid[0] = 0;
        mustReject(invalid, root.string());
        invalid.resize(12);
        mustReject(invalid, root.string());
        invalid = original;
        invalid[section(invalid, 3).sh_offset] ^= 1;
        mustReject(invalid, root.string());
        { std::ofstream file(source, std::ios::app); file << "\n// Changed SDK source\n"; }
        mustReject(original, root.string());
        std::filesystem::remove_all(root);
        std::cout << "PASS: module replacement, unchanged ABI/other kernels, malformed ELF, ABI/source rejection\n";
        return 0;
    } catch (const std::exception& error) {
        std::filesystem::remove_all(root);
        std::cerr << error.what() << '\n';
        return 1;
    }
}
