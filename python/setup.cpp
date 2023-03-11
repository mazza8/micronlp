#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "../micronlp/lm.h"
#include "../micronlp/metrics.h"

namespace py = pybind11;

PYBIND11_MODULE(micronlp, m) {
    py::class_<MLE>(m, "MLE")
        .def(py::init<int>())
        .def("fit", &MLE::fit, py::arg("corpus"))
        .def("perplexity", &MLE::perplexity, py::arg("ngrams"));

    m.def("edit_distance", &edit_distance,
                py::arg("source"), py::arg("target"),  py::arg("substitution_cost") = 1,
                "A function that computes the minimum edit distance.");
}