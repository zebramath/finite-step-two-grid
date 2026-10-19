#include "experiment/comparison_cases.hpp"
#include "experiment/reporting.hpp"
#include "multigrid/global_pcg.hpp"
#include "multigrid/two_grid_solver.hpp"
#include <array>
namespace {
void append_measurement(
    const experiment_support::ComparisonCase& item,
    const std::string& method, const std::string& parameter,
    const tgi::SparseMatrix& matrix, const tgi::Vector& rhs,
    const tgi::SparseMatrix& prolongation, int threads,
    experiment_support::Rows& rows) {
    const tgi::TwoGridCycle cycle(matrix, prolongation, 1, threads);
    const auto solved = tgi::solve_two_grid(
        rhs, cycle, 1.0e-6,
        experiment_support::maximum_two_grid_cycles);
    const double density =
        experiment_support::interpolation_density_percent(prolongation);
    rows.push_back({
        item.axis,
        std::to_string(item.fine),
        std::to_string(item.coarse),
        experiment_support::scientific(item.contrast, 0),
        item.field.name,
        method,
        parameter,
        experiment_support::fixed(density, 4),
        std::to_string(solved.cycles),
        tgi::stationary_status_name(solved.status),
        std::to_string(cycle.coarse_matrix().nnz())});
}
}
int main() {
    const int threads = experiment_support::thread_count();
    const auto cases = experiment_support::comparison_cases();
    experiment_support::Rows rows;
    constexpr std::array<int, 3> denominators{4, 3, 2};
    for (std::size_t case_index = 0; case_index < cases.size(); ++case_index) {
        const auto& item = cases[case_index];
        experiment_support::progress(
            "finite-path comparison " + std::to_string(case_index + 1U) +
            "/" + std::to_string(cases.size()) + ": " + item.field.name);
        const auto config = experiment_support::comparison_config(item);
        const auto grid = experiment_support::make_grid(config);
        const auto problem = experiment_support::make_problem(
            grid, item.field, config);
        const auto initial = tgi::build_geometric_interpolation(grid);
        tgi::GlobalEnergyPcgPath path(
            grid, problem.matrix, initial, threads);
        for (int denominator : denominators) {
            const int steps = experiment_support::fixed_path_steps(
                item.fine, denominator);
            path.advance_to(steps);
            const auto prolongation = path.prolongation();
            const std::string method = "finite-1/" + std::to_string(denominator);
            append_measurement(
                item, method,
                "m=" + std::to_string(steps),
                problem.matrix, problem.rhs, prolongation, threads,
                rows);
        }
        tgi::GlobalEnergyPcgPath endpoint_path(
            grid, problem.matrix, initial, threads);
        endpoint_path.advance_until_relative_residual(1.0e-10);
        const auto endpoint = endpoint_path.prolongation();
        append_measurement(
            item, "energy-endpoint", "column relres<=1e-10",
            problem.matrix, problem.rhs, endpoint, threads,
            rows);
    }
    experiment_support::save_csv("experiment1",
        {"axis", "N_h", "N_H", "contrast", "topology", "method", "parameter",
         "P_density_percent", "cycles", "status", "Ac_nnz"}, rows);
    return 0;
}
