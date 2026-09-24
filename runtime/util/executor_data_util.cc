// Copyright 2025 The ODML Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "runtime/util/executor_data_util.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <optional>
#include <utility>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/status_macros.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "litert/cc/litert_layout.h"  // from @litert
#include "litert/cc/litert_macros.h"  // from @litert
#include "litert/cc/litert_ranked_tensor_type.h"  // from @litert
#include "litert/cc/litert_tensor_buffer.h"  // from @litert
#include "runtime/executor/llm_executor_io_types.h"
#include "runtime/util/status_macros.h"  // IWYU pragma: keep
#include "runtime/util/tensor_buffer_util.h"

namespace litert::lm {
namespace {

absl::StatusOr<const ::litert::TensorBuffer*> GetEmbeddingsPtr(
    const ExecutorVisionData& data) {
  return data.GetEmbeddingsPtr();
}

absl::StatusOr<const ::litert::TensorBuffer*> GetEmbeddingsPtr(
    const ExecutorAudioData& data) {
  return data.GetProjectedAudioEmbeddingsPtr();
}

absl::StatusOr<::litert::TensorBuffer*> GetMutableEmbeddingsPtr(
    ExecutorVisionData& data) {
  return data.GetMutableEmbeddingsPtr();
}

absl::StatusOr<::litert::TensorBuffer*> GetMutableEmbeddingsPtr(
    ExecutorAudioData& data) {
  return data.GetMutableProjectedAudioEmbeddingsPtr();
}

// Concatenates host tensors shaped [1, N_i, ...] along the token axis (dim 1)
// into a single [1, sum(N_i), ...] tensor. Returns nullopt if no tensor is
// present, and an error if only some of them are present.
absl::StatusOr<std::optional<TensorBuffer>> ConcatenateAlongTokens(
    const std::vector<const TensorBuffer*>& tensors) {
  int present = 0;
  for (const auto* tensor : tensors) present += tensor != nullptr;
  if (present == 0) return std::nullopt;
  if (present != tensors.size()) {
    return absl::InvalidArgumentError(
        "Either all or none of the vision data must carry this tensor.");
  }
  LITERT_ASSIGN_OR_RETURN(auto first_type, tensors[0]->TensorType());
  ABSL_ASSIGN_OR_RETURN(auto dims, TensorBufferDims(*tensors[0]));
  if (dims.size() < 2) {
    return absl::InvalidArgumentError("Expected a token axis at dim 1.");
  }
  int total_tokens = 0;
  size_t total_size = 0;
  for (const auto* tensor : tensors) {
    ABSL_ASSIGN_OR_RETURN(auto tensor_dims, TensorBufferDims(*tensor));
    total_tokens += tensor_dims[1];
    LITERT_ASSIGN_OR_RETURN(size_t size, tensor->PackedSize());
    total_size += size;
  }
  dims[1] = total_tokens;
  ::litert::RankedTensorType combined_type(
      first_type.ElementType(),
      Layout(Dimensions(dims.begin(), dims.end())));
  LITERT_ASSIGN_OR_RETURN(
      auto combined, TensorBuffer::CreateManagedHostMemory(combined_type,
                                                           total_size));
  LITERT_ASSIGN_OR_RETURN(auto combined_lock,
                          ::litert::TensorBufferScopedLock::Create(
                              combined, TensorBuffer::LockMode::kWrite));
  char* dst = static_cast<char*>(combined_lock.second);
  for (const auto* tensor : tensors) {
    LITERT_ASSIGN_OR_RETURN(size_t size, tensor->PackedSize());
    LITERT_ASSIGN_OR_RETURN(
        auto lock, ::litert::TensorBufferScopedLock::Create(
                       const_cast<TensorBuffer&>(*tensor),
                       TensorBuffer::LockMode::kRead));
    memcpy(dst, lock.second, size);
    dst += size;
  }
  return std::optional<TensorBuffer>(std::move(combined));
}

// Combines the DeepStack features and M-RoPE offsets of several images.
absl::Status CombineVisionExtras(
    const std::vector<ExecutorVisionData>& executor_data,
    ExecutorVisionData& combined) {
  std::vector<const TensorBuffer*> deepstack;
  std::vector<const TensorBuffer*> mrope_offsets;
  for (const auto& data : executor_data) {
    auto ds = data.GetDeepstackEmbeddingsPtr();
    deepstack.push_back(ds.ok() ? *ds : nullptr);
    auto offsets = data.GetMropeOffsetsPtr();
    mrope_offsets.push_back(offsets.ok() ? *offsets : nullptr);
  }
  ABSL_ASSIGN_OR_RETURN(auto combined_deepstack,
                        ConcatenateAlongTokens(deepstack));
  combined.SetDeepstackEmbeddings(std::move(combined_deepstack));
  ABSL_ASSIGN_OR_RETURN(auto combined_offsets,
                        ConcatenateAlongTokens(mrope_offsets));
  combined.SetMropeOffsets(std::move(combined_offsets));
  return absl::OkStatus();
}

template <typename T>
absl::StatusOr<T> CombineExecutorDataImpl(std::vector<T>& executor_data) {
  if (executor_data.empty()) {
    return absl::InvalidArgumentError("Executor data is empty.");
  }
  if (executor_data.size() == 1) {
    // If there is only one image, we can just move it to the combined image
    // data.
    return std::move(executor_data[0]);
  }
  // If there are multiple executor data, we need to first combine them into a
  // TensorBuffer, then create a single ExecutorVisionData from the
  // TensorBuffer.
  int num_executor_data = executor_data.size();
  ABSL_ASSIGN_OR_RETURN(const auto* first_tensor,
                        GetEmbeddingsPtr(executor_data[0]));
  LITERT_ASSIGN_OR_RETURN(auto first_tensor_type, first_tensor->TensorType());
  ABSL_ASSIGN_OR_RETURN(auto first_tensor_dims,
                        TensorBufferDims(*first_tensor));
  int total_token_num = 0;
  int total_packed_size = 0;
  std::vector<int> combined_token_num;
  for (const auto& executor_data : executor_data) {
    ABSL_ASSIGN_OR_RETURN(const auto* embeddings_ptr,
                          GetEmbeddingsPtr(executor_data));
    ABSL_ASSIGN_OR_RETURN(auto dims, TensorBufferDims(*embeddings_ptr));
    if (dims.size() != 3 && dims.size() != 4) {
      return absl::InvalidArgumentError(
          "The embedding tensor type must have 3 or 4 dimensions.");
    }
    combined_token_num.push_back(dims[dims.size() - 2]);
    total_token_num += dims[dims.size() - 2];
    LITERT_ASSIGN_OR_RETURN(size_t packed_size, embeddings_ptr->PackedSize());
    total_packed_size += packed_size;
  }
  Layout combined_layout;
  if constexpr (std::is_same_v<T, ExecutorAudioData>) {
    combined_layout = Layout(Dimensions(
        {first_tensor_dims[0], total_token_num, first_tensor_dims[2]}));
  } else if (first_tensor_dims.size() == 3) {
    combined_layout = Layout(Dimensions(
        {first_tensor_dims[0], 1, total_token_num, first_tensor_dims[2]}));
  } else if (first_tensor_dims.size() == 4) {
    combined_layout =
        Layout(Dimensions({first_tensor_dims[0], first_tensor_dims[1],
                           total_token_num, first_tensor_dims[3]}));
  }
  ::litert::RankedTensorType combined_tensor_type(
      first_tensor_type.ElementType(), std::move(combined_layout));

  LITERT_ASSIGN_OR_RETURN(auto combined_tensor_buffer,
                          TensorBuffer::CreateManagedHostMemory(
                              combined_tensor_type, total_packed_size));
  LITERT_ASSIGN_OR_RETURN(
      auto combined_embeddings_lock_and_addr,
      ::litert::TensorBufferScopedLock::Create(combined_tensor_buffer,
                                               TensorBuffer::LockMode::kWrite));
  char* combined_tensor_buffer_ptr =
      static_cast<char*>(combined_embeddings_lock_and_addr.second);
  for (int i = 0; i < num_executor_data; ++i) {
    ABSL_ASSIGN_OR_RETURN(auto embeddings_ptr,
                          GetMutableEmbeddingsPtr(executor_data[i]));
    LITERT_ASSIGN_OR_RETURN(auto embeddings_size, embeddings_ptr->PackedSize());
    LITERT_ASSIGN_OR_RETURN(
        auto embeddings_lock_and_addr,
        ::litert::TensorBufferScopedLock::Create(
            *embeddings_ptr, TensorBuffer::LockMode::kRead));
    memcpy(combined_tensor_buffer_ptr, embeddings_lock_and_addr.second,
           embeddings_size);
    combined_tensor_buffer_ptr += embeddings_size;
  }
  if constexpr (std::is_same_v<T, ExecutorVisionData>) {
    ExecutorVisionData combined(std::move(combined_tensor_buffer),
                                /*per_layer_embeddings=*/std::nullopt);
    ABSL_RETURN_IF_ERROR(CombineVisionExtras(executor_data, combined));
    return combined;
  } else if constexpr (std::is_same_v<T, ExecutorAudioData>) {
    int num_audio_tokens = 0;
    for (const auto& executor_data : executor_data) {
      num_audio_tokens += executor_data.GetValidTokens();
    }
    return ExecutorAudioData(std::move(combined_tensor_buffer),
                             /*per_layer_embeddings=*/std::nullopt,
                             num_audio_tokens);
  } else {
    return absl::InvalidArgumentError("Executor data type is not supported.");
  }
}

}  // namespace

absl::StatusOr<ExecutorVisionData> CombineExecutorVisionData(
    std::vector<ExecutorVisionData>& executor_data) {
  return CombineExecutorDataImpl(executor_data);
}

absl::StatusOr<ExecutorAudioData> CombineExecutorAudioData(
    std::vector<ExecutorAudioData>& executor_data) {
  return CombineExecutorDataImpl(executor_data);
}

}  // namespace litert::lm
