#include <torch/extension.h>

#include <string>

#include "linear8/linear8_torch.h"


std::string hello() {
    return "Hello from I-Whisper C++ extension!";
}




PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "hello",
        &hello,
        "I-Whisper C++ extension smoke test"
    );
    m.def(
        "linear8_scalar",
        &linear8_scalar_torch,
        "I-Whisper Linear8 scalar kernel"
    );
}