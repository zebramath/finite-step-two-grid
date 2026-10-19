#pragma once
#include "core/linear_algebra.hpp"
#include <cstdint>
namespace tgi {
class StructuredGrid {
public:
    StructuredGrid(int fine_interior_points, int coarsening_ratio);
    int fine_n() const { return fine_n_; }
    int ratio() const { return ratio_; }
    int intervals() const { return fine_n_ + 1; }
    int fine_size() const { return fine_n_ * fine_n_; }
    int coarse_size() const { return coarse_n_ * coarse_n_; }
    double h() const { return 1.0 / static_cast<double>(intervals()); }
    int fine_id(int ix, int iy) const { return iy * fine_n_ + ix; }
    std::pair<int, int> fine_coords(int id) const;
    int coarse_id(int cx, int cy) const { return cy * coarse_n_ + cx; }
    std::pair<int, int> coarse_coords(int id) const;
    int coarse_fine_id(int coarse_id) const;
    bool is_coarse_node(int fine_id) const;
    std::vector<int> all_f_nodes() const;
private:
    int fine_n_;
    int ratio_;
    int coarse_n_;
};
enum class CoefficientDistribution {
    ChannelizedBinary,
    MeanderingChannelBinary,
    DiagonalChannelsBinary,
    ParallelChannelsBinary,
    BranchingChannelsBinary,
    WindingRingBinary
};
struct CoefficientOptions {
    CoefficientDistribution distribution =
        CoefficientDistribution::ChannelizedBinary;
    double contrast = 1e4;
    std::uint64_t seed = 1;
    int channel_background_block_size = 8;
    int channel_width_fine_cells = 2;
};
inline StructuredGrid::StructuredGrid(int fine_interior_points, int coarsening_ratio)
    : fine_n_(fine_interior_points), ratio_(coarsening_ratio), coarse_n_(0) {
    coarse_n_ = (fine_n_ + 1) / ratio_ - 1;
}
inline std::pair<int, int> StructuredGrid::fine_coords(int id) const {
    return {id % fine_n_, id / fine_n_};
}
inline std::pair<int, int> StructuredGrid::coarse_coords(int id) const {
    return {id % coarse_n_, id / coarse_n_};
}
inline int StructuredGrid::coarse_fine_id(int id) const {
    const auto [cx, cy] = coarse_coords(id);
    return fine_id((cx + 1) * ratio_ - 1, (cy + 1) * ratio_ - 1);
}
inline bool StructuredGrid::is_coarse_node(int id) const {
    const auto [ix, iy] = fine_coords(id);
    return ((ix + 1) % ratio_ == 0) && ((iy + 1) % ratio_ == 0);
}
inline std::vector<int> StructuredGrid::all_f_nodes() const {
    std::vector<int> nodes;
    nodes.reserve(static_cast<std::size_t>(fine_size() - coarse_size()));
    for (int fine = 0; fine < fine_size(); ++fine) {
        if (!is_coarse_node(fine)) nodes.push_back(fine);
    }
    return nodes;
}
namespace diffusion_problem_detail {
constexpr double pi = 3.141592653589793238462643383279502884;
inline double harmonic(double lhs, double rhs) {
    return 2.0 * lhs * rhs / (lhs + rhs);
}
inline std::uint64_t mix_bits(std::uint64_t value) {
    value += 0x9e3779b97f4a7c15ULL;
    value = (value ^ (value >> 30U)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27U)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31U);
}
inline bool is_high_conductivity_channel(double x, double y,
                                  double width) {
    const double horizontal_center =
        9.0 / 32.0 + 0.007 * std::sin(4.0 * pi * x + 0.30);
    const double vertical_center =
        23.0 / 32.0 + 0.007 * std::sin(3.0 * pi * y + 0.80);
    const bool horizontal =
        std::abs(y - horizontal_center) <= 0.5 * width;
    const bool vertical =
        std::abs(x - vertical_center) <= 0.5 * width;
    return horizontal || vertical;
}
inline bool is_high_conductivity_topology(
    CoefficientDistribution distribution, double x, double y,
    double width, std::uint64_t seed) {
    if (distribution == CoefficientDistribution::ChannelizedBinary) {
        return is_high_conductivity_channel(x, y, width);
    }
    const double phase = 2.0 * pi *
        static_cast<double>(mix_bits(seed) & 0xffffULL) / 65536.0;
    if (distribution == CoefficientDistribution::MeanderingChannelBinary) {
        const double center = 0.50 + 0.18 *
            std::sin(2.0 * pi * x + phase);
        return std::abs(y - center) <= 0.5 * width;
    }
    if (distribution == CoefficientDistribution::DiagonalChannelsBinary) {
        const double shift = 0.10 * std::sin(phase);
        return std::abs(y - x - shift) <= 0.5 * width ||
            std::abs(y - (1.0 - x) + shift) <= 0.5 * width;
    }
    if (distribution == CoefficientDistribution::ParallelChannelsBinary) {
        const double wobble = 0.015 * std::sin(4.0 * pi * x + phase);
        return std::abs(y - (0.25 + wobble)) <= 0.5 * width ||
            std::abs(y - (0.50 - wobble)) <= 0.5 * width ||
            std::abs(y - (0.75 + wobble)) <= 0.5 * width;
    }
    if (distribution == CoefficientDistribution::BranchingChannelsBinary) {
        const double trunk = 0.50 + 0.04 *
            std::sin(2.0 * pi * x + phase);
        const double branch_origin = 0.42;
        const double branch_distance = std::max(0.0, x - branch_origin);
        const double branch_wobble = 0.018 *
            std::sin(5.0 * pi * x + phase);
        const double upper = 0.50 + 0.72 * branch_distance + branch_wobble;
        const double lower = 0.50 - 0.72 * branch_distance - branch_wobble;
        return std::abs(y - trunk) <= 0.5 * width ||
            (x >= branch_origin &&
             (std::abs(y - upper) <= 0.5 * width ||
              std::abs(y - lower) <= 0.5 * width));
    }
    const double dx = x - 0.50;
    const double dy = y - 0.50;
    const double angle = std::atan2(dy, dx);
    const double radius = std::sqrt(dx * dx + dy * dy);
    const double target = 0.29 + 0.035 *
        std::sin(5.0 * angle + phase);
    return std::abs(radius - target) <= 0.5 * width;
}
}
inline Vector make_coefficient(const StructuredGrid& grid,
                                  const CoefficientOptions& options) {
    Vector values(static_cast<std::size_t>(grid.fine_size()));
    const double width =
        static_cast<double>(options.channel_width_fine_cells) * grid.h();
    for (int id = 0; id < grid.fine_size(); ++id) {
        const auto [ix, iy] = grid.fine_coords(id);
        const double x = static_cast<double>(ix + 1) * grid.h();
        const double y = static_cast<double>(iy + 1) * grid.h();
        const int block_x =
            (ix + 1) / options.channel_background_block_size;
        const int block_y =
            (iy + 1) / options.channel_background_block_size;
        const std::uint64_t hash = diffusion_problem_detail::mix_bits(
            options.seed ^
            (static_cast<std::uint64_t>(
                 static_cast<std::uint32_t>(block_x)) << 32U) ^
            static_cast<std::uint64_t>(
                static_cast<std::uint32_t>(block_y)));
        const bool high_inclusion = (hash & 1ULL) != 0ULL;
        values[static_cast<std::size_t>(id)] =
            (diffusion_problem_detail::is_high_conductivity_topology(
                 options.distribution, x, y, width, options.seed) ||
             high_inclusion)
                ? options.contrast
                : 1.0;
    }
    return values;
}
inline SparseMatrix assemble_diffusion(const StructuredGrid& grid,
                                const Vector& coefficient) {
    const double inverse_h2 = 1.0 / (grid.h() * grid.h());
    std::vector<int> row_ptr(
        static_cast<std::size_t>(grid.fine_size()) + 1U, 0);
    std::vector<int> col_idx;
    Vector values;
    col_idx.reserve(static_cast<std::size_t>(5 * grid.fine_size()));
    values.reserve(static_cast<std::size_t>(5 * grid.fine_size()));
    for (int id = 0; id < grid.fine_size(); ++id) {
        const auto [ix, iy] = grid.fine_coords(id);
        const double center =
            coefficient[static_cast<std::size_t>(id)];
        double diagonal = 0.0;
        const auto face_value = [&](int neighbor) {
            return diffusion_problem_detail::harmonic(
                center, coefficient[static_cast<std::size_t>(neighbor)]);
        };
        const auto add_neighbor = [&](int neighbor) {
            const double face = face_value(neighbor);
            col_idx.push_back(neighbor);
            values.push_back(-face * inverse_h2);
            diagonal += face * inverse_h2;
        };
        if (iy > 0) add_neighbor(id - grid.fine_n());
        else diagonal += center * inverse_h2;
        if (ix > 0) add_neighbor(id - 1);
        else diagonal += center * inverse_h2;
        col_idx.push_back(id);
        values.push_back(0.0);
        const std::size_t diagonal_position = values.size() - 1U;
        if (ix + 1 < grid.fine_n()) add_neighbor(id + 1);
        else diagonal += center * inverse_h2;
        if (iy + 1 < grid.fine_n()) add_neighbor(id + grid.fine_n());
        else diagonal += center * inverse_h2;
        values[diagonal_position] = diagonal;
        row_ptr[static_cast<std::size_t>(id) + 1U] =
            static_cast<int>(values.size());
    }
    return SparseMatrix(
        grid.fine_size(), grid.fine_size(), std::move(row_ptr),
        std::move(col_idx), std::move(values));
}
}
