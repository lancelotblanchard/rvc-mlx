#include "weights.h"

#include <sstream>
#include <stdexcept>

namespace rvc::detail {

const mx::array& Weights::at(const std::string& key) const {
    auto it = arrays.find(key);
    if (it == arrays.end()) throw std::runtime_error("'" + path + "' has no tensor '" + key + "'");
    return it->second;
}

std::string format(const mx::Shape& shape) {
    std::ostringstream s;
    s << "(";
    for (size_t i = 0; i < shape.size(); ++i) s << (i ? ", " : "") << shape[i];
    s << ")";
    return s.str();
}

Weights loadWeights(const std::string& path, const std::string& expectedKind, mx::Dtype dtype) {
    Weights w;
    w.path = path;
    try {
        auto loaded = mx::load_safetensors(path);
        w.meta = std::move(loaded.second);
        for (auto& [name, arr] : loaded.first) {
            mx::array a = mx::issubdtype(arr.dtype(), mx::floating) ? mx::astype(arr, dtype) : arr;
            w.arrays.emplace(name, a);
        }
    } catch (const std::exception& e) {
        throw std::runtime_error("Could not read '" + path + "': " + e.what());
    }
    auto kind = w.meta.find("kind");
    if (kind != w.meta.end() && kind->second != expectedKind)
        throw std::runtime_error("'" + path + "' is a " + kind->second + " file, expected " + expectedKind);
    std::vector<mx::array> all;
    all.reserve(w.arrays.size());
    for (auto& [_, a] : w.arrays) all.push_back(a);
    mx::eval(all);
    return w;
}

}  // namespace rvc::detail
