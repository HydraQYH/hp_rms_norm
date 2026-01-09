#include <Python.h>
#include <torch/all.h>

extern "C" {
  /* Creates a dummy empty _C module that can be imported from Python.
     The import from Python will load the .so consisting of this file
     in this extension, so that the TORCH_LIBRARY static initializers
     below are run. */
  PyObject* PyInit__C(void)
  {
      static struct PyModuleDef module_def = {
          PyModuleDef_HEAD_INIT,
          "_C",   /* name of module */
          NULL,   /* module documentation, may be NULL */
          -1,     /* size of per-interpreter state of the module,
                     or -1 if the module keeps state in global variables. */
          NULL,   /* methods */
      };
      return PyModule_Create(&module_def);
  }
}

// Declaration
void rms_norm(
    torch::Tensor& output,
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& residual,
    double eps);

// Defines the operators
TORCH_LIBRARY(hp_rms_norm, m) {
  m.def("rms_norm(Tensor output, Tensor input, Tensor weight, Tensor residual, float eps) -> ()");
}

TORCH_LIBRARY_IMPL(hp_rms_norm, CUDA, m) {
  m.impl("rms_norm", &rms_norm);
}
