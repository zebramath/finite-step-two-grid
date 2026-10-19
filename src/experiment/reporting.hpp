#pragma once
#include "core/linear_algebra.hpp"
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
namespace experiment_support {
using Row = std::vector<std::string>;
using Rows = std::vector<Row>;
inline std::string fixed(double value, int precision = 3) {
    std::ostringstream stream;
    stream << std::fixed << std::setprecision(precision) << value;
    return stream.str();
}
inline std::string scientific(double value, int precision = 3) {
    std::ostringstream stream;
    stream << std::scientific << std::setprecision(precision) << value;
    return stream.str();
}
inline void progress(const std::string& message) { std::cerr << message << '\n'; }
inline double interpolation_density_percent(const tgi::SparseMatrix& p) {
    return 100.0 * static_cast<double>(p.nnz()) /
        (static_cast<double>(p.rows()) * static_cast<double>(p.cols()));
}
inline std::filesystem::path results_directory() {
    const char* directory = std::getenv("TGI_RESULTS_DIR");
    const std::filesystem::path path = directory == nullptr ? TGI_RESULTS_DIR : directory;
    std::filesystem::create_directories(path);
    return path;
}
inline void save_csv(const std::string& name, const Row& headers, const Rows& rows) {
    const auto path = results_directory() / (name + ".csv");
    std::ofstream stream(path);
    const auto write = [&](const Row& row) {
        for (std::size_t i = 0; i < row.size(); ++i) stream << (i == 0 ? "" : ",") << row[i];
        stream << '\n';
    };
    write(headers);
    for (const auto& row : rows) write(row);
    if (!stream) throw std::runtime_error("Cannot write " + path.string());
}
}
