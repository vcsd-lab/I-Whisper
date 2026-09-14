#pragma once

#ifdef IWHISPER_HAS_NEON

#include <arm_neon.h>

#else

#ifndef SIMDE_ENABLE_NATIVE_ALIASES
#define SIMDE_ENABLE_NATIVE_ALIASES
#endif

#include <simde/arm/neon.h>

#endif
