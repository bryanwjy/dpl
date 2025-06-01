/* Copyright 2023-2025 Bryan Wong */

#pragma once

#if defined(__x86_64__) || defined(_M_X64)
#  define DPL_ARCH_x86_64 1
#  define DPL_ARCH_x86 1
#  define DPL_ARCH_WIDTH 64
#elif defined(__aarch64__) || defined(_M_ARM64)
#  define DPL_ARCH_AARCH64 1
#  define DPL_ARCH_ARM 1
#  define DPL_ARCH_WIDTH 64
#elif defined(i386) || defined(__i386__) || defined(__i386) || defined(_M_IX86)
#  error "Unsupported Architecture"
#elif defined(__ARM_ARCH_2__)
#  error "Unsupported Architecture"
#elif defined(__ARM_ARCH_3__) || defined(__ARM_ARCH_3M__)
#  error "Unsupported Architecture"
#elif defined(__ARM_ARCH_4T__) || defined(__TARGET_ARM_4T)
#  error "Unsupported Architecture"
#elif defined(__ARM_ARCH_5_) || defined(__ARM_ARCH_5E_)
#  error "Unsupported Architecture"
#elif defined(__ARM_ARCH_6T2_) || defined(__ARM_ARCH_6T2_)
#  error "Unsupported Architecture"
#elif defined(__ARM_ARCH_6__) || defined(__ARM_ARCH_6J__) || defined(__ARM_ARCH_6K__) || \
    defined(__ARM_ARCH_6Z__) || defined(__ARM_ARCH_6ZK__)
#  error "Unsupported Architecture"
#elif defined(__ARM_ARCH_7__) || defined(__ARM_ARCH_7A__) || defined(__ARM_ARCH_7R__) || \
    defined(__ARM_ARCH_7M__) || defined(__ARM_ARCH_7S__)
#  error "Unsupported Architecture"
#elif defined(__ARM_ARCH_7A__) || defined(__ARM_ARCH_7R__) || defined(__ARM_ARCH_7M__) || \
    defined(__ARM_ARCH_7S__)
#  error "Unsupported Architecture"
#elif defined(__ARM_ARCH_7R__) || defined(__ARM_ARCH_7M__) || defined(__ARM_ARCH_7S__)
#  error "Unsupported Architecture"
#elif defined(__ARM_ARCH_7M__)
#  error "Unsupported Architecture"
#elif defined(__ARM_ARCH_7S__)
#  error "Unsupported Architecture"
#elif defined(mips) || defined(__mips__) || defined(__mips)
#  error "Unsupported Architecture"
#elif defined(__sh__)
#  error "Unsupported Architecture"
#elif defined(__powerpc) || defined(__powerpc__) || defined(__powerpc64__) || \
    defined(__POWERPC__) || defined(__ppc__) || defined(__PPC__) || defined(_ARCH_PPC)
#  error "Unsupported Architecture"
#elif defined(__PPC64__) || defined(__ppc64__) || defined(_ARCH_PPC64)
#  error "Unsupported Architecture"
#elif defined(__sparc__) || defined(__sparc)
#  error "Unsupported Architecture"
#elif defined(__m68k__)
#  error "Unsupported Architecture"
#else
#  error "Unknown Architecture"
#endif
