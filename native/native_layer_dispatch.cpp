// PDScope native decode dispatch: submit -> wait -> stage -> merge in one call.
//
// Per decode layer the Python runtime used to: build a jobs dict, pack the rows,
// allocate ctypes arrays, build one future per expert, submit to the author's
// native queue, join that queue, slice the block per expert, copy the block to
// the GPU and scatter it. Measured on the two-socket host that sequence costs
// 200-400 us per layer, i.e. a large share of the MoE tail that the next layer's
// attention has to wait for.
//
// This module performs the whole tail natively:
//   1. fill the packed input (decode repeats one activation row per expert),
//   2. submit one packed task to the author's CPUInfer queue and join it,
//   3. copy the resulting block to the GPU,
//   4. scatter it into the layer output with the merge coefficients.
// The arithmetic is unchanged: the same author kernel writes the same rows into
// the same buffer, and the scatter accumulates in the declared order.
#define PY_SSIZE_T_CLEAN
#include <torch/extension.h>

#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>

#include <cstring>
#include <vector>

namespace {

void check_cuda(cudaError_t status, const char* what) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
    }
}

void fill_packed(intptr_t packed_ptr, intptr_t row_ptr, int64_t rows, int64_t row_bytes) {
    auto* dst = reinterpret_cast<char*>(packed_ptr);
    auto* src = reinterpret_cast<const char*>(row_ptr);
    for (int64_t r = 0; r < rows; ++r) {
        std::memcpy(dst + r * row_bytes, src, static_cast<size_t>(row_bytes));
    }
}

// submit_and_merge(pool, layer, row_ptr, rows, row_bytes, packed, block, scratch,
//                  count, selected, coeff, ids, out, tokens, weights)
//   pool/layer   author CPUInfer queue and loaded native layer object
//   row_ptr      one activation row (rows x row_bytes copied from it)
//   packed/block host buffers, each rows x hidden BF16
//   scratch      device buffer, rows x hidden BF16
//   out/tokens/weights  device merge targets
void submit_group(py::object pool, py::object layer, intptr_t row_ptr, int64_t rows,
                  int64_t row_bytes, intptr_t packed_ptr, intptr_t block_ptr,
                  intptr_t count_ptr, intptr_t selected_ptr, intptr_t coeff_ptr,
                  std::vector<int64_t> ids) {
    if (rows < 1 || row_bytes < 2 || row_bytes % 2 != 0) {
        throw std::runtime_error("invalid decode pack shape");
    }
    if (static_cast<int64_t>(ids.size()) != rows) {
        throw std::runtime_error("expert id count must match the packed row count");
    }
    auto* selected = reinterpret_cast<int64_t*>(selected_ptr);
    auto* coeff = reinterpret_cast<float*>(coeff_ptr);
    auto* count = reinterpret_cast<int32_t*>(count_ptr);
    for (int64_t i = 0; i < rows; ++i) {
        selected[i] = ids[static_cast<size_t>(i)];
        coeff[i] = 1.0f;
    }
    *count = static_cast<int32_t>(rows);
    fill_packed(packed_ptr, row_ptr, rows, row_bytes);
    py::object task = layer.attr("forward_task")(count_ptr, 1, selected_ptr, coeff_ptr, packed_ptr,
                                                 block_ptr, false);
    pool.attr("submit")(task);
}

torch::Tensor finish_and_merge(py::object pool, intptr_t block_ptr, int64_t rows, int64_t row_bytes,
                               torch::Tensor scratch, torch::Tensor out, torch::Tensor tokens,
                               torch::Tensor weights) {
    if (!scratch.is_cuda() || !out.is_cuda() || !tokens.is_cuda() || !weights.is_cuda()) {
        throw std::runtime_error("scratch/out/tokens/weights must live on the CUDA device");
    }
    pool.attr("sync")();
    if (scratch.numel() * scratch.element_size() != rows * row_bytes) {
        throw std::runtime_error("scratch buffer must match the packed block");
    }
    auto stream = at::cuda::getCurrentCUDAStream(out.get_device());
    check_cuda(cudaMemcpyAsync(scratch.data_ptr(), reinterpret_cast<void*>(block_ptr),
                               static_cast<size_t>(rows) * static_cast<size_t>(row_bytes),
                               cudaMemcpyHostToDevice, stream.stream()),
               "cudaMemcpyAsync(stage)");
    out.index_add_(0, tokens, scratch * weights.unsqueeze(1));
    return out;
}

torch::Tensor submit_and_merge(py::object pool, py::object layer, intptr_t row_ptr, int64_t rows,
                               int64_t row_bytes, intptr_t packed_ptr, intptr_t block_ptr,
                               torch::Tensor scratch, intptr_t count_ptr, intptr_t selected_ptr,
                               intptr_t coeff_ptr, std::vector<int64_t> ids, torch::Tensor out,
                               torch::Tensor tokens, torch::Tensor weights) {
    if (rows < 1 || row_bytes < 2 || row_bytes % 2 != 0) {
        throw std::runtime_error("invalid decode pack shape");
    }
    if (static_cast<int64_t>(ids.size()) != rows) {
        throw std::runtime_error("expert id count must match the packed row count");
    }
    if (!scratch.is_cuda() || !out.is_cuda() || !tokens.is_cuda() || !weights.is_cuda()) {
        throw std::runtime_error("scratch/out/tokens/weights must live on the CUDA device");
    }
    auto* selected = reinterpret_cast<int64_t*>(selected_ptr);
    auto* coeff = reinterpret_cast<float*>(coeff_ptr);
    auto* count = reinterpret_cast<int32_t*>(count_ptr);
    for (int64_t i = 0; i < rows; ++i) {
        selected[i] = ids[static_cast<size_t>(i)];
        coeff[i] = 1.0f;
    }
    *count = static_cast<int32_t>(rows);
    fill_packed(packed_ptr, row_ptr, rows, row_bytes);
    // One packed author call for the whole layer group.
    py::object task = layer.attr("forward_task")(count_ptr, 1, selected_ptr, coeff_ptr, packed_ptr,
                                                 block_ptr, false);
    pool.attr("submit")(task);
    pool.attr("sync")();
    // Stage and merge on the caller's stream.
    if (scratch.numel() * scratch.element_size() != rows * row_bytes) {
        throw std::runtime_error("scratch buffer must match the packed block");
    }
    auto stream = at::cuda::getCurrentCUDAStream(out.get_device());
    check_cuda(cudaMemcpyAsync(scratch.data_ptr(), reinterpret_cast<void*>(block_ptr),
                               static_cast<size_t>(rows) * static_cast<size_t>(row_bytes),
                               cudaMemcpyHostToDevice, stream.stream()),
               "cudaMemcpyAsync(stage)");
    out.index_add_(0, tokens, scratch * weights.unsqueeze(1));
    return out;
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("submit_and_merge", &submit_and_merge,
          "Native decode MoE tail: submit, join, stage and merge in one call");
    m.def("submit_group", &submit_group,
          "Pack and submit the decode CPU group to the native queue without joining it");
    m.def("finish_and_merge", &finish_and_merge,
          "Join the native queue, stage the packed block and scatter it into the output");
}
