// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#if !DPL_MODULES
#  include "dpl/core/type_traits/common_size_type.h"
#endif

__DPL_DEFAULT_NAMESPACE_BEGIN

namespace datapar {
namespace atom {
template <typename T, typename U>
concept common_size_with =
    sizeof(T) == sizeof(U) && sizeof(common_size_type_t<T, U>) == sizeof(T) &&
    sizeof(common_size_type_t<T, U>) == sizeof(U);
}

template <typename T, typename U>
concept common_size_with =
    requires {
        typename common_size_type_t<T, U>;
        typename common_size_type_t<U, T>;
    } && same_as<common_size_type_t<U, T>, common_size_type_t<T, U>> &&
    atom::common_size_with<T, U> && atom::common_size_with<U, T>;

} // namespace datapar

__DPL_DEFAULT_NAMESPACE_END
