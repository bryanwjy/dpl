// Copyright 2026 Bryan Wong
module;

export module mpfr.interface;
import dpl;

export namespace mpfr {
enum class result_flag : unsigned char {
    none = 0,
    inexact = 1 << 0,
    underflow = 1 << 1,
    overflow = 1 << 2,
    invalid = 0xff,
};

constexpr result_flag operator|(result_flag flag, result_flag val) noexcept {
    return static_cast<result_flag>(
        dpl::to_underlying(flag) | dpl::to_underlying(val));
}
constexpr result_flag operator&(result_flag flag, result_flag val) noexcept {
    return static_cast<result_flag>(
        dpl::to_underlying(flag) & dpl::to_underlying(val));
}
constexpr result_flag& operator|=(result_flag& flag, result_flag val) noexcept {
    return flag = flag | val;
}
constexpr result_flag& operator&=(result_flag& flag, result_flag val) noexcept {
    return flag = flag | val;
}

template <typename T>
struct storage {
    static_assert(dpl::is_trivially_copyable_v<T>);
    using element_type =
        dpl::conditional_t<dpl::is_array_v<T>, dpl::remove_extent_t<T>, T>;
    using data_type = dpl::conditional_t<dpl::is_array_v<T>,
        unsigned char[sizeof(element_type)][dpl::extent_v<T>],
        unsigned char[sizeof(element_type)]>;

    static constexpr dpl::size_t extent = []() {
        if constexpr (dpl::is_array_v<T>) {
            return dpl::extent_v<T>;
        } else {
            return 1zu;
        }
    }();

    data_type data;

    constexpr storage() = default;
    constexpr storage(storage const&) = default;
    constexpr ~storage() = default;
    explicit constexpr storage(T const& from) noexcept
        : storage(dpl::bit_cast<storage>(from)) {}
    constexpr storage& operator=(T const& from) noexcept {
        return *this = dpl::bit_cast<storage>(from);
    }

    template <dpl::size_t I>
    constexpr element_type get() const noexcept
    requires dpl::is_array_v<T>
    {
        return dpl::bit_cast<T>(*this[I]);
    }

    constexpr element_type get() const noexcept
    requires (!dpl::is_array_v<T>)
    {
        return dpl::bit_cast<T>(*this);
    }
    constexpr element_type operator*() const noexcept
    requires (!dpl::is_array_v<T>)
    {
        return dpl::bit_cast<T>(*this);
    }
};

template <typename T>
explicit storage(T const&) -> storage<T>;

enum class expectation_type : unsigned char {
    f16,
    bf16,
    f32,
    f64,
};

template <dpl::floating_point_like T>
struct expectation {
    T value;
    float residual;
    result_flag flags;
};

template <dpl::floating_point_like T, dpl::size_t N>
struct data_entry {
    storage<dpl::conditional_t<N == 1, T, T[N]>> input;
    storage<T> value;
    storage<float> residual;
    storage<result_flag> flags;
};

template <dpl::uint16 V>
struct file_header;

struct file_header_base {
    static constexpr dpl::uint16 latest_version = 1;
    static constexpr dpl::uint32 magic_value = 0xeafc69b7;

    storage<dpl::uint32> magic;
    storage<dpl::uint16> version;
    storage<dpl::uint16> header_size;
};

using latest_file_header = file_header<file_header_base::latest_version>;

template <>
struct file_header<1> {
    file_header_base base = {
        .magic = storage{file_header_base::magic_value},
        .version = storage<dpl::uint16>{1},
        .header_size = storage<dpl::uint16>{128},
    };
    storage<expectation_type> expectation_type;
    storage<dpl::uint32> record_offset;
    storage<dpl::uint32> record_count;
    storage<dpl::uint32> function_name_offset;
    storage<dpl::uint32> function_name_length;
    storage<dpl::uint8> function_arity;
    unsigned char reserved[102];
};

static_assert(
    sizeof(file_header<1>) == file_header<1>{}.base.header_size.get());

} // namespace mpfr
