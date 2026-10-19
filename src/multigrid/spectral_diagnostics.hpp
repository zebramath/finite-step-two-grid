#pragma once
#include "multigrid/two_grid_solver.hpp"
#include <cstdint>
#include <limits>
#include <random>
namespace tgi {
struct SpectralRadiusEstimate {
    double rho = 1.0;
    double stage_difference = 0.0;
};
namespace spectral_diagnostics_detail {
inline double energy_inner_product(
    const SparseMatrix& a, const Vector& left, const Vector& right,
    Vector& action) {
    a.multiply(right, action);
    return dot(left, action);
}
inline double energy_norm(
    const SparseMatrix& a, const Vector& value, Vector& action) {
    return std::sqrt(std::max(
        0.0, energy_inner_product(a, value, value, action)));
}
inline double largest_tridiagonal_eigenvalue(
    const std::vector<double>& diagonal,
    const std::vector<double>& off_diagonal) {
    double scale = 0.0;
    for (double value : diagonal) scale = std::max(scale, std::abs(value));
    for (double value : off_diagonal) scale = std::max(scale, std::abs(value));
    if (scale == 0.0) return 0.0;
    double lower = std::numeric_limits<double>::infinity();
    double upper = -std::numeric_limits<double>::infinity();
    for (std::size_t row = 0; row < diagonal.size(); ++row) {
        double radius = 0.0;
        if (row > 0U) radius += std::abs(off_diagonal[row - 1U] / scale);
        if (row < off_diagonal.size()) {
            radius += std::abs(off_diagonal[row] / scale);
        }
        lower = std::min(lower, diagonal[row] / scale - radius);
        upper = std::max(upper, diagonal[row] / scale + radius);
    }
    const double pivot_floor = 16.0 * std::numeric_limits<double>::epsilon();
    for (int iteration = 0; iteration < 96; ++iteration) {
        const double point = 0.5 * (lower + upper);
        int count_below = 0;
        double pivot = diagonal.front() / scale - point;
        if (pivot < 0.0) ++count_below;
        if (std::abs(pivot) < pivot_floor) pivot = -pivot_floor;
        for (std::size_t row = 1; row < diagonal.size(); ++row) {
            const double edge = off_diagonal[row - 1U] / scale;
            pivot = diagonal[row] / scale - point - edge * edge / pivot;
            if (pivot < 0.0) ++count_below;
            if (std::abs(pivot) < pivot_floor) pivot = -pivot_floor;
        }
        if (count_below < static_cast<int>(diagonal.size())) {
            lower = point;
        } else {
            upper = point;
        }
    }
    return scale * (0.5 * (lower + upper));
}
}
inline SpectralRadiusEstimate estimate_two_grid_spectral_radius(
    const SparseMatrix& a, const TwoGridCycle& cycle,
    std::uint64_t seed = 0x6a09e667f3bcc909ULL,
    int krylov_iterations = 240) {
    std::mt19937_64 generator(seed);
    std::uniform_int_distribution<int> sign(0, 1);
    Vector basis(static_cast<std::size_t>(a.rows()));
    for (double& entry : basis) {
        entry = sign(generator) == 0 ? -1.0 : 1.0;
    }
    Vector action;
    double scale = spectral_diagnostics_detail::energy_norm(
        a, basis, action);
    for (double& entry : basis) entry /= scale;
    const Vector zero(static_cast<std::size_t>(a.rows()), 0.0);
    Vector residual;
    TwoGridCycle::Workspace workspace;
    Vector basis_action = action;
    for (double& entry : basis_action) entry /= scale;
    Vector previous(static_cast<std::size_t>(a.rows()), 0.0);
    Vector image(static_cast<std::size_t>(a.rows()));
    Vector previous_action(static_cast<std::size_t>(a.rows()), 0.0);
    double previous_beta = 0.0;
    std::vector<double> diagonal;
    std::vector<double> off_diagonal;
    diagonal.reserve(static_cast<std::size_t>(krylov_iterations));
    off_diagonal.reserve(static_cast<std::size_t>(krylov_iterations - 1));
    SpectralRadiusEstimate result;
    double halfway_rho = std::numeric_limits<double>::quiet_NaN();
    const int halfway = krylov_iterations / 2;
    for (int iteration = 1; iteration <= krylov_iterations; ++iteration) {
        image = basis;
        cycle.apply(zero, image, residual, workspace);
        const double alpha = dot(basis_action, image);
        diagonal.push_back(alpha);
        axpy(-alpha, basis, image);
        if (iteration > 1) axpy(-previous_beta, previous, image);
        const double current_component = dot(basis_action, image);
        axpy(-current_component, basis, image);
        if (iteration > 1) {
            const double previous_component = dot(previous_action, image);
            axpy(-previous_component, previous, image);
        }
        if (iteration == halfway) {
            halfway_rho =
                spectral_diagnostics_detail::largest_tridiagonal_eigenvalue(
                    diagonal, off_diagonal);
        }
        if (iteration == krylov_iterations) break;
        const double beta = spectral_diagnostics_detail::energy_norm(
            a, image, action);
        if (!(beta > 0.0) || !std::isfinite(beta)) break;
        off_diagonal.push_back(beta);
        previous.swap(basis);
        basis.swap(image);
        previous_action.swap(basis_action);
        basis_action = action;
        for (double& entry : basis) entry /= beta;
        for (double& entry : basis_action) entry /= beta;
        previous_beta = beta;
    }
    result.rho =
        spectral_diagnostics_detail::largest_tridiagonal_eigenvalue(
            diagonal, off_diagonal);
    result.stage_difference = std::isfinite(halfway_rho)
        ? std::abs(result.rho - halfway_rho) : 0.0;
    return result;
}
}
