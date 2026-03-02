# Ops Status (MLX -> MIL)

- Supported: **124**
- Not yet implemented: **37**
- Not supported: **7**

## Supported

| MLX Op | MIL Op |
| --- | --- |
| `add` | `add` |
| `addmm` | `addmm` |
| `all` | `all` |
| `allclose` | `allclose` |
| `any` | `any` |
| `arange` | `arange` |
| `arccos` | `acos` |
| `arcsin` | `asin` |
| `arctan` | `atan` |
| `arctanh` | `atanh` |
| `argmax` | `reduce_argmax` |
| `argmin` | `reduce_argmin` |
| `array_equal` | `array_equal` |
| `astype` | `cast` |
| `atleast_1d` | `atleast_1d` |
| `atleast_2d` | `atleast_2d` |
| `atleast_3d` | `atleast_3d` |
| `bitwisebinary` | `bitwisebinary` |
| `broadcast` | `broadcast_to` |
| `broadcast_arrays` | `broadcast_arrays` |
| `broadcast_to` | `broadcast_to` |
| `cast` | `cast` |
| `concatenate` | `concat` |
| `const` | `const` |
| `constant` | `const` |
| `contiguous` | `identity` |
| `conv1d` | `conv` |
| `conv2d` | `conv` |
| `conv3d` | `conv` |
| `conv_general` | `conv` |
| `conv_transpose1d` | `conv_transpose` |
| `conv_transpose2d` | `conv_transpose` |
| `conv_transpose3d` | `conv_transpose` |
| `copy` | `identity` |
| `degrees` | `degrees` |
| `diag` | `diag` |
| `diagonal` | `diagonal` |
| `divide` | `real_div` |
| `divmod` | `divmod` |
| `expm1` | `expm1` |
| `eye` | `eye` |
| `flatten` | `flatten` |
| `floor_div` | `floor_div` |
| `floor_divide` | `floor_div` |
| `full` | `full` |
| `full_like` | `full_like` |
| `gather` | `gather` |
| `greater` | `greater` |
| `inner` | `inner` |
| `inverse` | `inverse` |
| `isclose` | `isclose` |
| `isfinite` | `isfinite` |
| `isinf` | `isinf` |
| `isnan` | `isnan` |
| `isneginf` | `isneginf` |
| `isposinf` | `isposinf` |
| `kron` | `kron` |
| `less` | `less` |
| `linspace` | `linspace` |
| `log10` | `log10` |
| `log1p` | `log1p` |
| `log2` | `log2` |
| `logaddexp` | `logaddexp` |
| `logsumexp` | `reduce_log_sum_exp` |
| `matmul` | `matmul` |
| `max` | `reduce_max` |
| `maximum` | `maximum` |
| `mean` | `reduce_mean` |
| `meshgrid` | `meshgrid` |
| `min` | `reduce_min` |
| `mod` | `mod` |
| `moveaxis` | `moveaxis` |
| `mul` | `mul` |
| `multiply` | `mul` |
| `nan_to_num` | `nan_to_num` |
| `negative` | `negative` |
| `number_of_elements` | `number_of_elements` |
| `ones` | `ones` |
| `ones_like` | `ones_like` |
| `outer` | `outer` |
| `pow` | `pow` |
| `power` | `pow` |
| `prod` | `reduce_prod` |
| `radians` | `radians` |
| `real_div` | `real_div` |
| `reciprocal` | `inverse` |
| `reduce_argmax` | `reduce_argmax` |
| `reduce_argmin` | `reduce_argmin` |
| `reduce_max` | `reduce_max` |
| `reduce_mean` | `reduce_mean` |
| `reduce_min` | `reduce_min` |
| `reduce_prod` | `reduce_prod` |
| `reduce_sum` | `reduce_sum` |
| `remainder` | `mod` |
| `reshape` | `reshape` |
| `rmsnorm` | `rmsnorm` |
| `rope` | `rope` |
| `scaled_dot_product_attention` | `scaled_dot_product_attention` |
| `scaleddotproductattention` | `scaled_dot_product_attention` |
| `select` | `select` |
| `sigmoid` | `sigmoid` |
| `slice` | `slice_by_index` |
| `slice_by_index` | `slice_by_index` |
| `softmax` | `softmax` |
| `squeeze` | `squeeze` |
| `std` | `std` |
| `stop_gradient` | `identity` |
| `sub` | `sub` |
| `subtract` | `sub` |
| `sum` | `reduce_sum` |
| `swapaxes` | `swapaxes` |
| `take` | `gather` |
| `take_along_axis` | `gather_along_axis` |
| `tensordot` | `tensordot` |
| `trace` | `trace` |
| `transpose` | `transpose` |
| `tri` | `tri` |
| `tril` | `tril` |
| `triu` | `triu` |
| `unflatten` | `unflatten` |
| `var` | `var` |
| `where` | `select` |
| `zeros` | `zeros` |
| `zeros_like` | `zeros_like` |

## Not Yet Implemented

| MLX Op | Proposed MIL Lowering |
| --- | --- |
| `arccosh` | `log/sqrt formula` |
| `arcsinh` | `log/sqrt formula` |
| `arctan2` | `atan + quadrant correction` |
| `argpartition` | `no direct op; approximate via sort/topk` |
| `as_strided` | `no safe MIL equivalent` |
| `bitwise_and` | `no direct op (emulate expensive)` |
| `bitwise_invert` | `no direct op (emulate expensive)` |
| `bitwise_or` | `no direct op (emulate expensive)` |
| `bitwise_xor` | `no direct op (emulate expensive)` |
| `block_masked_mm` | `custom op or decompose with major perf loss` |
| `conjugate` | `complex dtype gap` |
| `cummax` | `while_loop scan + max state` |
| `cummin` | `while_loop scan + min state` |
| `cumprod` | `while_loop + running state` |
| `erfinv` | `numeric approx (poly/rational)` |
| `gather_mm` | `fused gather+matmul (custom likely)` |
| `hadamard_transform` | `structured matmul / staged butterflies` |
| `hamming` | `arange + cos formula` |
| `hanning` | `arange + cos formula` |
| `imag` | `complex dtype gap` |
| `left_shift` | `bitwise shift gap` |
| `logcumsumexp` | `scan op missing; while_loop decomposition` |
| `masked_scatter` | `scatter with mask semantics mismatch` |
| `median` | `partition/sort needed; expensive` |
| `partition` | `partition primitive missing` |
| `put_along_axis` | `scatter update semantics mismatch` |
| `real` | `complex dtype gap` |
| `repeat` | `tile + reshape + slice` |
| `right_shift` | `bitwise shift gap` |
| `roll` | `slice + concat` |
| `scatter_add` | `scatter reduce(add) missing` |
| `scatter_add_axis` | `scatter reduce(add) missing` |
| `scatter_max` | `scatter reduce(max) missing` |
| `scatter_min` | `scatter reduce(min) missing` |
| `scatter_prod` | `scatter reduce(prod) missing` |
| `segmented_mm` | `segmented/fused matmul primitive missing` |
| `sort` | `argsort + gather_along_axis` |

## Not Supported

| MLX Op | Reason |
| --- | --- |
| `depends` | Control-dependency semantics are not representable in the current MIL lowering model. |
| `from_fp8` | **Policy: not supported.** Non-`fp16`/`fp32` quantization path is out of scope for this project. |
| `gather_qmm` | **Policy: not supported.** Quantized qmm-family op is out of scope for this project. |
| `qqmm` | **Policy: not supported.** Quantized qmm-family op is out of scope for this project. |
| `quantized_matmul` | **Policy: not supported.** Non-`fp16`/`fp32` quantization path is out of scope for this project. |
| `to_fp8` | **Policy: not supported.** Non-`fp16`/`fp32` quantization path is out of scope for this project. |
| `view` | Stride-aware view semantics are not representable in MIL (outside contiguous reshape-compatible cases). |
