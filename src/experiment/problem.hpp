#pragma once
#include "pde/diffusion_problem.hpp"
#include <cstdlib>
#include <string>
namespace experiment_support {
inline constexpr int maximum_two_grid_cycles = 20000;
inline int thread_count() {
    const char* value = std::getenv("TGI_THREADS");
    return value == nullptr ? 4 : std::max(1, std::stoi(value));
}
inline int fixed_path_steps(int fine_intervals, int denominator) {
    return std::max(
        1, (fine_intervals + denominator / 2) / denominator);
}
struct BasicConfig {
    int fine_intervals = 128;
    int coarse_intervals = 16;
    int threads = thread_count();
    double contrast = 1.0e4;
};
inline tgi::StructuredGrid make_grid(const BasicConfig& config) {
    return tgi::StructuredGrid(
        config.fine_intervals - 1,
        config.fine_intervals / config.coarse_intervals);
}
struct FieldCase {
    std::string name;
    tgi::CoefficientDistribution distribution;
};
inline const std::vector<FieldCase>& channel_topologies() {
    static const std::vector<FieldCase> fields{
        {"cross-channel", tgi::CoefficientDistribution::ChannelizedBinary},
        {"meandering-channel",
         tgi::CoefficientDistribution::MeanderingChannelBinary},
        {"diagonal-channels",
         tgi::CoefficientDistribution::DiagonalChannelsBinary},
        {"parallel-channels",
         tgi::CoefficientDistribution::ParallelChannelsBinary},
        {"branching-channels",
         tgi::CoefficientDistribution::BranchingChannelsBinary},
        {"winding-ring",
         tgi::CoefficientDistribution::WindingRingBinary}
    };
    return fields;
}
struct ExperimentProblem {
    tgi::SparseMatrix matrix;
    tgi::Vector rhs;
};
inline ExperimentProblem make_problem(
    const tgi::StructuredGrid& grid, const FieldCase& field,
    const BasicConfig& config, std::uint64_t seed = 1) {
    tgi::CoefficientOptions options;
    options.distribution = field.distribution;
    options.contrast = config.contrast;
    options.seed = seed;
    options.channel_background_block_size = grid.ratio();
    options.channel_width_fine_cells = 2;
    auto coefficient = tgi::make_coefficient(grid, options);
    auto matrix = tgi::assemble_diffusion(grid, coefficient);
    tgi::Vector rhs(static_cast<std::size_t>(grid.fine_size()), 1.0);
    return {std::move(matrix), std::move(rhs)};
}
}
