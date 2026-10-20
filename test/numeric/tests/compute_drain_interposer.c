/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdatomic.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

static atomic_int armed;
static atomic_int allocation_armed;

void arm_compute_drain_failure(void) { atomic_store(&armed, 1); }
void arm_scratch_allocation_failure(void) { atomic_store(&allocation_armed, 1); }

static void *real_symbol(void *handle, const char *name) {
  void *(*lookup)(void *, const char *) =
      (void *(*)(void *, const char *))dlvsym(RTLD_NEXT, "dlsym", "GLIBC_2.2.5");
  if (!lookup)
    lookup =
        (void *(*)(void *, const char *))dlvsym(RTLD_NEXT, "dlsym", "GLIBC_2.17");
  if (!lookup)
    abort();
  return lookup(handle, name);
}

int hipMalloc(void **pointer, size_t bytes) {
  if (bytes == 512 && atomic_exchange(&allocation_armed, 0))
    return 2; /* hipErrorOutOfMemory, without poisoning the stream. */
  int (*allocate)(void **, size_t) =
      (int (*)(void **, size_t))real_symbol(RTLD_NEXT, "hipMalloc");
  if (!allocate)
    abort();
  return allocate(pointer, bytes);
}

// ORC searches explicit ROCm library handles before the process image. Route
// only this symbol through the injector even for those handle-scoped lookups.
void *dlsym(void *handle, const char *name) {
  if (strcmp(name, "hipMalloc") == 0)
    return (void *)&hipMalloc;
  return real_symbol(handle, name);
}

int hipStreamSynchronize(void *stream) {
  if (atomic_load(&armed))
    return 999; /* hipErrorUnknown: no destructive fault is launched on the GPU. */
  int (*synchronize)(void *) =
      (int (*)(void *))dlsym(RTLD_NEXT, "hipStreamSynchronize");
  if (!synchronize)
    abort();
  return synchronize(stream);
}
