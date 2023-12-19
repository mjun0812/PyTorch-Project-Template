#include "DCNv3/dcnv3.h"

namespace pytorch_cpp {
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("dcnv3_forward", &dcnv3_forward, "dcnv3_forward");
  m.def("dcnv3_backward", &dcnv3_backward, "dcnv3_backward");
}
}  // namespace pytorch_cpp
