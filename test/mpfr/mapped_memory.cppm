// Copyright 2026 Bryan Wong
module;
#define DPL_MODULES 1
#include "dpl/config.h"

#include <fcntl.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

export module mpfr:mapped_memory;
import dpl;

export namespace mpfr {

class mapped_memory {
    enum file_descriptor_t : int {
        invalid = -1,
    };
    static constexpr size_t npos = -1zu;
    struct storage {
        storage() = default;
        explicit storage(file_descriptor_t descriptor) noexcept
            : size([=] {
                struct stat file_stat{};
                if (descriptor == invalid ||
                    fstat(descriptor, &file_stat) == -1) {
                    return npos;
                }

                return static_cast<size_t>(file_stat.st_size);
            }())
            , data([=](size_t size) {
                if (size != npos && size > 0) {
                    static constexpr int unused = 0;
                    void* ptr = mmap(nullptr, size, PROT_READ, MAP_PRIVATE,
                        descriptor, unused);
                    return ptr == MAP_FAILED ? nullptr : ptr;
                }

                return static_cast<void*>(nullptr);
            }(this->size)) {
            ::close(descriptor);
        }

        size_t size = npos;
        void* data = nullptr;
    };

public:
    constexpr mapped_memory() noexcept = default;
    explicit mapped_memory(char const* file_path) noexcept
        : storage_(open_file(file_path)) {}

    mapped_memory(mapped_memory const&) = delete;
    mapped_memory& operator=(mapped_memory const&) = delete;

    mapped_memory(mapped_memory&& other) noexcept
        : storage_{dpl::exchange(other.storage_, storage())} {}

    mapped_memory& operator=(mapped_memory&& other) noexcept {
        this->unmap();
        storage_ = dpl::exchange(other.storage_, storage());
        return *this;
    }

    DPL_NODISCARD
    constexpr size_t size() const noexcept { return storage_.size; }

    DPL_NODISCARD
    constexpr void const* data() const noexcept { return storage_.data; }

    constexpr void unmap() noexcept {
        if (storage_.data != nullptr) {
            auto to_unmap = dpl::exchange(storage_, storage());
            munmap(to_unmap.data, to_unmap.size);
        }
    }

    DPL_NODISCARD
    explicit constexpr operator bool() const noexcept {
        return storage_.data != nullptr;
    }

    constexpr ~mapped_memory() { this->unmap(); }

private:
    DPL_NODISCARD
    static file_descriptor_t open_file(char const* path) noexcept {
        return static_cast<file_descriptor_t>(::open(path, O_RDONLY));
    }

    storage storage_;
};

bool is_regular_file(char const* path) noexcept {
    struct stat path_stat;

    // stat returns 0 on success, -1 on failure (e.g., file doesn't exist)
    if (stat(path, &path_stat) != 0) {
        return false;
    }

    // S_ISREG evaluates to true if it is a regular file
    return S_ISREG(path_stat.st_mode);
}

FILE* create_copy(char const* dst, char const* src) noexcept {
    mapped_memory const src_mem(src);
    if (!src_mem) {
        return nullptr;
    }

    struct dst_file_t {
        FILE* file;
        ~dst_file_t() {
            if (file != nullptr) {
                fclose(file);
            }
        }

        explicit operator bool() const { return file != nullptr; }
    } dst_file{.file = fopen(dst, "wb+")};

    if (!dst_file) {
        return nullptr;
    }

    setvbuf(dst_file.file, nullptr, _IONBF, 0);

    if (fwrite(src_mem.data(), 1zu, src_mem.size(), dst_file.file) <
        src_mem.size()) {
        return nullptr;
    }

    setvbuf(dst_file.file, nullptr, _IONBF, BUFSIZ);
    fseek(dst_file.file, 0, SEEK_SET);

    return dpl::exchange(dst_file.file, nullptr);
}
} // namespace mpfr
