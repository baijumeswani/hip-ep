/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#pragma once

#include "hip/Compiler/RocMlirArtifactTarget.h"
#include "hip/env.h"
#include "morphizen/morphizen.hpp"
#include "morphizen/util.hpp"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Config/llvm-config.h"
#include "llvm/Support/SHA256.h"
#include <algorithm>
#include <array>
#include <cstring>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl_lite.h>
#include <map>
#include <string>

#ifdef HIPDNN_EP_LINK_HIP_HOST
#include <hip/hip_runtime_api.h>
#endif

extern "C" const unsigned char runtime_bc_data[];
extern "C" const std::size_t runtime_bc_data_size;
extern "C" const char *morphizen_get_build_info();

namespace mlir_compilation {

using ArtifactCompatibility = std::map<std::string, std::string>;

inline std::string artifactSha256(const void *data, size_t size) {
  const auto bytes =
      llvm::ArrayRef<uint8_t>(static_cast<const uint8_t *>(data), size);
  return llvm::toHex(llvm::SHA256::hash(bytes), true);
}

inline std::string artifactFileSha256(morphizen::FileReader &reader) {
  reader.rewind();
  llvm::SHA256 hash;
  std::array<uint8_t, 65536> buffer;
  size_t remaining = reader.size();
  while (remaining != 0) {
    const size_t count =
        reader.fread(buffer.data(), std::min(remaining, buffer.size()));
    if (count == 0 || count > remaining) {
      throw morphizen::ArtifactCompatibilityError(
          "Truncated HIP artifact dependency while computing its identity.");
    }
    hash.update(llvm::ArrayRef<uint8_t>(buffer.data(), count));
    remaining -= count;
  }
  return llvm::toHex(hash.final(), true);
}

inline bool artifactOnlyOption(const std::string &key) {
  return key == "prebuild_cache_context" || key == "cache_key" ||
         key == "cacheKey" || key == "cache_dir" || key == "config_file" ||
         key == "dump_dir";
}

inline ArtifactCompatibility
artifactCompatibility(const morphizen::PassContext &context) {
  ArtifactCompatibility result{
      {"manifest_version", "1"},
      {"runtime_abi", "3"},
      {"compiler_build",
       artifactSha256(morphizen_get_build_info(),
                      std::strlen(morphizen_get_build_info()))},
      {"llvm_version", LLVM_VERSION_STRING},
      {"runtime_sha256", artifactSha256(runtime_bc_data, runtime_bc_data_size)},
      {"rocmlir_target", hip::compiler::rocMlirArtifactTarget()},
      {"optimization_level",
       context.get_provider_option("optimization_level", "2")},
      {"artifact_format",
       context.get_provider_option("artifact_format", "LLVM_IR")}};
  for (const auto &[key, value] : context.get_all_provider_options()) {
    if (!artifactOnlyOption(key) && key != "optimization_level" &&
        key != "artifact_format") {
      result["provider." + key] = value;
    }
  }
  auto config = context.get_config_proto();
  for (auto it = config.mutable_provider_options()->begin();
       it != config.mutable_provider_options()->end();) {
    if (artifactOnlyOption(it->first)) {
      it = config.mutable_provider_options()->erase(it);
    } else {
      ++it;
    }
  }
  std::string config_bytes;
  {
    google::protobuf::io::StringOutputStream stream(&config_bytes);
    google::protobuf::io::CodedOutputStream coded(&stream);
    coded.SetSerializationDeterministic(true);
    if (!config.SerializeToCodedStream(&coded)) {
      throw morphizen::ArtifactCompatibilityError(
          "Cannot serialize HIP compiler configuration identity.");
    }
  }
  result["compiler_config_sha256"] =
      artifactSha256(config_bytes.data(), config_bytes.size());
  // These switches affect generated code, precision, dispatch or retained
  // workspace. Debug/profiling/output-path switches do not affect identity.
  for (const auto *key :
       {"HIPDNN_EP_PIPELINE", "HIPDNN_EP_HIPSR", "HIPDNN_EP_ROCMLIR",
        "HIPDNN_EP_ROCMLIR_AUTOTUNE", "ROCK_ARCH", "HIP_CUSTOM_KERNELS_ARCH",
        "HIPDNN_EP_BUFFERIZE_COPY_BEFORE_WRITE", "HIPDNN_EP_MATMUL_DP4A",
        "HIPDNN_EP_MATMUL_DP4A_MAX_M", "HIPDNN_EP_AUTOTUNE"}) {
    result[std::string("env.") + key] = hipdnn_ep::env_string(key);
  }
#ifdef HIPDNN_EP_LINK_HIP_HOST
  int device = 0;
  int runtime = 0;
  hipDeviceProp_t properties{};
  if (hipGetDevice(&device) != hipSuccess ||
      hipGetDeviceProperties(&properties, device) != hipSuccess ||
      hipRuntimeGetVersion(&runtime) != hipSuccess) {
    throw morphizen::ArtifactCompatibilityError(
        "Cannot establish the selected HIP device/runtime identity.");
  }
  result["gpu_isa"] = properties.gcnArchName;
  result["hip_runtime_version"] = std::to_string(runtime);
#else
  result["gpu_isa"] = "no-hip-host";
#endif
  return result;
}

inline void
validateArtifactCompatibility(const ArtifactCompatibility &recorded,
                              const ArtifactCompatibility &requested) {
  for (const auto &[key, value] : requested) {
    const auto it = recorded.find(key);
    if (it == recorded.end() || it->second != value) {
      throw morphizen::ArtifactCompatibilityError(
          "HIP artifact compatibility mismatch for '" + key +
          "'; regenerate the artifact with the requested compiler/runtime "
          "and options.");
    }
  }
  if (recorded.size() != requested.size()) {
    throw morphizen::ArtifactCompatibilityError(
        "HIP artifact has an unsupported compatibility manifest; regenerate "
        "it with the current compiler.");
  }
}

} // namespace mlir_compilation
