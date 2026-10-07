# XSeg SAME padding repair (2026-10-08)

The converted generic XSeg checkpoint uses the original XSeg architecture.
This repair changes the PyTorch compatibility layer's transposed convolution
alignment, not the architecture or checkpoint. TensorFlow is not a runtime
dependency.

For a 3x3, stride-2 SAME transposed convolution, the previous symmetric padding
and output padding produced the expected size but shifted values up and left.
The decoder's six upsampling stages consequently misaligned the converted
weights. The repaired operation computes an unpadded transpose, then removes
the forward SAME padding, with the extra odd pixel on the bottom/right.
Kernels smaller than the stride use output padding to reach the requested size.
Bias, weight scaling and NCHW/NHWC handling are preserved. Other padding modes
retain their previous behavior. The ME network architecture does not change.

## Evidence

- 31 regression tests: a nonzero integer origin fixture and 30 combinations
  of kernel sizes 1-5, strides 1-3, and both layouts. Output and input/weight/bias
  gradients are checked against an independently computed forward-convolution
  adjoint, using non-square images and nonzero bias.
- The pinned generic inference checkpoint remains SHA256
  `26e45677ef3136e0327f0fd51e452cbea81a703ee7fea9121b01ba58da65c385`.
  It was not retrained or replaced with a local QA checkpoint.
- Before/after inspection covered all 78 tutorial aligned faces. The corrected
  masks were accepted visually. This is not a full TensorFlow/PyTorch network
  numerical equivalence claim or a mask-quality benchmark.
- All actual repository training inputs were refreshed: 482 loose aligned JPGs
  and 16 members in two explicitly used PAK/ZIP training archives. Original
  bytes were backed up; decoded pixels, non-mask metadata and manual polygons
  were verified unchanged. Historical evaluation fixtures were preserved.
- The existing 64px ME tutorial model resumed from iteration 5510 to 15110.
  A 600-second external guard requested save/close; completion ACK, saved
  metadata, finished heartbeat and process exit agreed on iteration 15110.
  The job exited normally with code 0; there were no live trainer processes
  at the later hard deadline. No extra training was run for visual quality.

Existing models must reset their sampling/data state when resuming after
aligned content changes; optimizer and model weights can be retained. The
tutorial resume used this option. Its short-run prediction remains a development
example, not a final identity or visual-quality acceptance.

## Sources

- [Original XSeg model](https://github.com/iperov/DeepFaceLab/blob/e4b7543ffa1d73b26fce1e31852727f658ba490c/core/leras/models/XSeg.py)
- [TensorFlow's documented SAME padding and transposed convolution semantics](https://github.com/tensorflow/tensorflow/blob/v2.16.1/tensorflow/python/ops/nn_ops.py)

The sources document the alignment being preserved; no TensorFlow package is
installed or invoked by the repaired production path.
