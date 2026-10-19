#include "experiment/problem.hpp"
#include "experiment/reporting.hpp"
#include "multigrid/global_pcg.hpp"
#include "multigrid/two_grid_solver.hpp"
namespace {
experiment_support::Row measure_until_converged(
    const std::string& method,
    const tgi::SparseMatrix& prolongation, const tgi::TwoGridCycle& cycle,
    const tgi::Vector& rhs) {
    const auto solved = tgi::solve_two_grid(rhs, cycle, 1.0e-6, 100000);
    if (solved.status != tgi::StationaryIterationStatus::Converged) {
        throw std::runtime_error("endpoint comparison missed the solve tolerance");
    }
    return {
        method,
        std::to_string(solved.cycles),
        experiment_support::fixed(solved.effective_factor, 9),
        experiment_support::fixed(experiment_support::interpolation_density_percent(prolongation), 6),
        std::to_string(cycle.coarse_matrix().nnz())};
}
}
int main() {
    const int threads = experiment_support::thread_count();
    experiment_support::BasicConfig config;
    const auto& field = experiment_support::channel_topologies().front();
    const tgi::StructuredGrid grid = experiment_support::make_grid(config);
    const auto problem = experiment_support::make_problem(
        grid, field, config);
    experiment_support::progress("geometric interpolation comparison");
    const auto geometric = tgi::build_geometric_interpolation(grid);
    const tgi::TwoGridCycle geometric_cycle(
        problem.matrix, geometric, 1, threads);
    const auto geometric_measurement = measure_until_converged(
        "geometric", geometric,
        geometric_cycle, problem.rhs);
    experiment_support::progress("energy-minimizing interpolation comparison");
    tgi::GlobalEnergyPcgPath endpoint_path(
        grid, problem.matrix, geometric, threads);
    endpoint_path.advance_until_relative_residual(1.0e-10);
    const auto energy = endpoint_path.prolongation();
    const tgi::TwoGridCycle energy_cycle(
        problem.matrix, energy, 1, threads);
    const auto energy_measurement = measure_until_converged(
        "energy-minimizing",
        energy, energy_cycle, problem.rhs);
    experiment_support::save_csv("experiment6", {"method", "cycles", "rho_eff", "P_density_percent", "Ac_nnz"},
        {geometric_measurement, energy_measurement});
    return 0;
}
