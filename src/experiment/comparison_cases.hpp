#pragma once
#include "experiment/problem.hpp"
namespace experiment_support {
struct ComparisonCase {
    std::string axis;
    int fine = 32;
    int coarse = 8;
    double contrast = 1.0e4;
    FieldCase field;
};
inline std::vector<ComparisonCase> comparison_cases() {
    const auto& topologies = channel_topologies();
    return {
        {"size", 32, 8, 1.0e4, topologies[0]},
        {"size", 64, 8, 1.0e4, topologies[0]},
        {"size", 64, 16, 1.0e4, topologies[0]},
        {"center", 128, 16, 1.0e4, topologies[0]},
        {"large-scale", 256, 16, 1.0e4, topologies[0]},
        {"large-scale", 256, 16, 1.0e4, topologies[5]},
        {"contrast", 128, 16, 1.0e2, topologies[0]},
        {"contrast", 128, 16, 1.0e6, topologies[0]},
        {"topology", 128, 16, 1.0e4, topologies[1]},
        {"topology", 128, 16, 1.0e4, topologies[2]},
        {"topology", 128, 16, 1.0e4, topologies[3]},
        {"topology", 128, 16, 1.0e4, topologies[4]},
        {"topology", 128, 16, 1.0e4, topologies[5]}};
}
inline BasicConfig comparison_config(const ComparisonCase& item) {
    BasicConfig config;
    config.fine_intervals = item.fine;
    config.coarse_intervals = item.coarse;
    config.contrast = item.contrast;
    return config;
}
}
