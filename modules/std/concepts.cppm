// Copyright 2025 Bryan Wong
module;

#define DPL_MODULES 1

#include "dpl/config.h"

export module dpl:std.concepts;
export import :std.stddef;
import :std.details.concepts;

__DPL_DEFAULT_NAMESPACE_BEGIN
// NOLINTBEGIN(misc-unused-using-decls)
using __DPL array_initializable;
using __DPL assignable_from;
using __DPL bool_constant_like;
using __DPL boolean_testable;
using __DPL common_reference_with;
using __DPL common_with;
using __DPL constructible_from;
using __DPL convertible_to;
using __DPL copy_constructible;
using __DPL copyable;
using __DPL core_convertible_to;
using __DPL default_initializable;
using __DPL derived_from;
using __DPL destructible;
using __DPL different_from;
using __DPL enumeration;
using __DPL equality_comparable;
using __DPL equality_comparable_with;
using __DPL explicitly_convertible_to;
using __DPL floating_point;
using __DPL index_sequence_like;
using __DPL integer_sequence_like;
using __DPL integral;
using __DPL integral_constant_like;
using __DPL invocable;
using __DPL movable;
using __DPL move_constructible;
using __DPL regular;
using __DPL regular_invocable;
using __DPL same_as;
using __DPL scoped_enumeration;
using __DPL semiregular;
using __DPL signed_integral;
using __DPL swappable;
using __DPL swappable_with;
using __DPL to_index_sequence;
using __DPL to_integer_sequence;
using __DPL totally_ordered;
using __DPL totally_ordered_with;
using __DPL tuple_like;
using __DPL unscoped_enumeration;
using __DPL unsigned_integral;

using __DPL ranges::get_element;
using __DPL ranges::swap;
// NOLINTEND(misc-unused-using-decls)

__DPL_DEFAULT_NAMESPACE_END
