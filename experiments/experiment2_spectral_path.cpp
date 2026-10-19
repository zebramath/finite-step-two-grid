#include "experiment/problem.hpp"
#include "experiment/reporting.hpp"
#include "multigrid/global_pcg.hpp"
#include "multigrid/spectral_diagnostics.hpp"
namespace {
constexpr int scan_cycle_limit = 12000;
void append_measurement(
    const std::string& topology, int steps,
    const tgi::SparseMatrix& matrix, const tgi::Vector& rhs,
    const tgi::SparseMatrix& prolongation, int threads,
    experiment_support::Rows& rows) {
    const tgi::TwoGridCycle cycle(matrix, prolongation, 1, threads);
    const auto spectral = tgi::estimate_two_grid_spectral_radius(
        matrix, cycle, 0x243f6a8885a308d3ULL,
        200);
    const auto solved = tgi::solve_two_grid(
        rhs, cycle, 1.0e-6, scan_cycle_limit);
    rows.push_back({topology, std::to_string(steps),
        experiment_support::fixed(spectral.rho, 10),
        experiment_support::scientific(spectral.stage_difference, 5),
        experiment_support::fixed(solved.effective_factor, 10),
        experiment_support::scientific(cycle.interpolation_energy(), 12),
        std::to_string(solved.cycles),
        experiment_support::scientific(solved.relative_residual, 8),
        tgi::stationary_status_name(solved.status)});
}
void scan(const experiment_support::BasicConfig& config,
          const experiment_support::FieldCase& field, experiment_support::Rows& rows) {
    const auto grid = experiment_support::make_grid(config);
    const auto problem = experiment_support::make_problem(grid, field, config);
    const auto initial = tgi::build_geometric_interpolation(grid);
    tgi::GlobalEnergyPcgPath path(grid, problem.matrix, initial, config.threads);
    for (int m = 1; m <= config.fine_intervals; ++m) {
        experiment_support::progress(field.name + " " + std::to_string(m));
        path.advance_to(m);
        append_measurement(field.name, m, problem.matrix, problem.rhs,
            path.prolongation(), config.threads, rows);
    }
    tgi::GlobalEnergyPcgPath endpoint(grid, problem.matrix, initial, config.threads);
    endpoint.advance_until_relative_residual(1e-10);
    append_measurement(field.name, -1, problem.matrix, problem.rhs,
        endpoint.prolongation(), config.threads, rows);
}
}
int main() {
    experiment_support::BasicConfig config;
    experiment_support::Rows rows;
    const auto& fields = experiment_support::channel_topologies();
    scan(config, fields[0], rows);
    scan(config, fields[5], rows);
    experiment_support::save_csv("experiment2", {"topology", "m", "rho_TG",
        "spectral_stage_difference", "rho_eff", "J",
        "cycles", "relative_residual", "status"}, rows);
    return 0;
}
