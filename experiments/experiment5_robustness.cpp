#include "experiment/problem.hpp"
#include "experiment/reporting.hpp"
#include "multigrid/global_pcg.hpp"
#include "multigrid/two_grid_solver.hpp"
#include <algorithm>
#include <array>
#include <chrono>
namespace {
using Clock = std::chrono::steady_clock;
struct TimingSample {
    double setup_s = 0.0;
    double solve_s = 0.0;
    int cycles = 0;
};
struct RhsCase {
    std::string name;
    tgi::Vector values;
};
enum class Method {
    Geometric,
    EnergyEndpoint,
    FiniteOneThird
};
double elapsed_seconds(Clock::time_point begin) {
    return std::chrono::duration<double>(
               Clock::now() - begin)
        .count();
}
tgi::Vector normalized(tgi::Vector values) {
    const double length = tgi::norm2(values);
    for (double& value : values) value /= length;
    return values;
}
std::vector<RhsCase> make_rhs_cases(const tgi::StructuredGrid& grid) {
    constexpr double pi = 3.141592653589793238462643383279502884;
    const std::size_t size = static_cast<std::size_t>(grid.fine_size());
    std::vector<RhsCase> cases;
    cases.push_back({"constant", normalized(tgi::Vector(size, 1.0))});
    for (const auto& specification : {
             std::pair<std::string, int>{"smooth-low", 0},
             {"smooth-mixed", 1},
             {"localized-left", 2},
             {"localized-right", 3},
             {"oscillatory", 4}}) {
        tgi::Vector rhs(size, 0.0);
        for (int node = 0; node < grid.fine_size(); ++node) {
            const auto [ix, iy] = grid.fine_coords(node);
            const double x = static_cast<double>(ix + 1) * grid.h();
            const double y = static_cast<double>(iy + 1) * grid.h();
            double value = 0.0;
            if (specification.second == 0) {
                value = std::sin(pi * x) * std::sin(pi * y);
            } else if (specification.second == 1) {
                value = std::sin(2.0 * pi * x) *
                        std::sin(3.0 * pi * y) +
                    0.35 * std::sin(5.0 * pi * x) *
                        std::sin(2.0 * pi * y);
            } else if (specification.second == 2) {
                const double dx = x - 0.28;
                const double dy = y - 0.72;
                value = std::exp(-90.0 * (dx * dx + dy * dy));
            } else if (specification.second == 3) {
                const double dx = x - 0.74;
                const double dy = y - 0.31;
                value = std::exp(-75.0 * (dx * dx + dy * dy));
            } else {
                value = std::sin(7.0 * pi * x) *
                    std::sin(6.0 * pi * y);
            }
            rhs[static_cast<std::size_t>(node)] = value;
        }
        cases.push_back({specification.first, normalized(std::move(rhs))});
    }
    return cases;
}
tgi::SparseMatrix make_transfer(
    const tgi::StructuredGrid& grid, const tgi::SparseMatrix& matrix,
    const tgi::SparseMatrix& initial, Method method, int threads) {
    if (method == Method::Geometric) return initial;
    tgi::GlobalEnergyPcgPath path(grid, matrix, initial, threads);
    if (method == Method::FiniteOneThird) {
        path.advance_to(experiment_support::fixed_path_steps(
            grid.intervals(), 3));
    } else {
        path.advance_until_relative_residual(1.0e-10);
    }
    return path.prolongation();
}
TimingSample timing_sample(
    const tgi::StructuredGrid& grid, const tgi::SparseMatrix& matrix,
    const tgi::Vector& rhs, Method method, int threads) {
    const auto begin = Clock::now();
    const auto initial = tgi::build_geometric_interpolation(grid);
    const auto transfer = make_transfer(
        grid, matrix, initial, method, threads);
    const tgi::TwoGridCycle cycle(matrix, transfer, 1, threads);
    const double setup_s = elapsed_seconds(begin);
    const auto solve_begin = Clock::now();
    const auto solved = tgi::solve_two_grid(rhs, cycle, 1.0e-6, 100000);
    const double solve_s = elapsed_seconds(solve_begin);
    if (solved.status != tgi::StationaryIterationStatus::Converged) throw std::runtime_error("timed solve did not converge");
    return {setup_s, solve_s, solved.cycles};
}
}
experiment_support::Rows timing() {
    const int threads = experiment_support::thread_count();
    const auto& field = experiment_support::channel_topologies().front();
    experiment_support::BasicConfig timing_config;
    const tgi::StructuredGrid timing_grid =
        experiment_support::make_grid(timing_config);
    const auto timing_problem = experiment_support::make_problem(
        timing_grid, field, timing_config);
    std::array<Method, 3> order{
        Method::Geometric, Method::EnergyEndpoint, Method::FiniteOneThird};
    for (Method method : order) {
        (void)timing_sample(
            timing_grid, timing_problem.matrix, timing_problem.rhs,
            method, threads);
    }
    std::array<std::vector<TimingSample>, 3> samples;
    int repetition = 0;
    do {
        experiment_support::progress(
            "timing repetition " + std::to_string(++repetition) + "/6");
        for (Method method : order) {
            samples[static_cast<std::size_t>(method)].push_back(timing_sample(
                timing_grid, timing_problem.matrix, timing_problem.rhs,
                method, threads));
        }
    } while (std::next_permutation(order.begin(), order.end()));
    experiment_support::Rows timing_samples;
    const std::array<std::string, 3> names{
        "geometric", "energy-endpoint", "finite-1/3"};
    for (std::size_t method = 0; method < names.size(); ++method) {
        for (std::size_t run = 0; run < samples[method].size(); ++run) {
            const auto& sample = samples[method][run];
            timing_samples.push_back({"timing", std::to_string(run + 1U), names[method],
                "", std::to_string(sample.cycles), "converged",
                experiment_support::fixed(sample.setup_s, 6),
                experiment_support::fixed(sample.solve_s, 6),
                experiment_support::fixed(sample.setup_s + sample.solve_s, 6)});
        }
    }
    return timing_samples;
}
void save(const experiment_support::Rows& seeds, const experiment_support::Rows& rhs,
          const experiment_support::Rows& timings) {
    experiment_support::Rows rows;
    for (const auto& r : seeds) rows.push_back({"seed", r[0], r[1], r[2], r[3], r[4], "", "", ""});
    for (const auto& r : rhs) rows.push_back({"rhs", r[0], r[1], r[2], r[3], r[4], "", "", ""});
    rows.insert(rows.end(), timings.begin(), timings.end());
    experiment_support::save_csv("experiment5", {"kind", "case", "method", "parameter", "cycles",
        "status", "setup_s", "solve_s", "total_s"}, rows);
}
int main(int argc, char** argv) {
    if (argc > 1 && std::string(argv[1]) == "--timing") {
        save({}, {}, timing());
        return 0;
    }
    const int threads = experiment_support::thread_count();
    constexpr int maximum_cycles =
        experiment_support::maximum_two_grid_cycles;
    constexpr std::array<std::uint64_t, 6> seeds{1, 3, 7, 11, 19, 23};
    const auto& field = experiment_support::channel_topologies().front();
    experiment_support::BasicConfig config;
    config.fine_intervals = 64;
    const tgi::StructuredGrid grid = experiment_support::make_grid(config);
    experiment_support::Rows seed_rows;
    for (std::uint64_t seed : seeds) {
        experiment_support::progress(
            "coefficient seed " + std::to_string(seed));
        const auto problem = experiment_support::make_problem(
            grid, field, config, seed);
        const auto initial = tgi::build_geometric_interpolation(grid);
        for (const auto& item : {
                 std::pair<std::string, Method>{
                     "finite-1/3", Method::FiniteOneThird},
                 {"energy-endpoint", Method::EnergyEndpoint}}) {
            const auto transfer = make_transfer(
                grid, problem.matrix, initial, item.second, threads);
            const tgi::TwoGridCycle cycle(
                problem.matrix, transfer, 1, threads);
            const auto solved = tgi::solve_two_grid(
                problem.rhs, cycle, 1.0e-6, maximum_cycles);
            seed_rows.push_back({
                std::to_string(seed), item.first,
                item.second == Method::FiniteOneThird
                    ? "m=" + std::to_string(
                          experiment_support::fixed_path_steps(
                              config.fine_intervals, 3))
                    : "relres<=1e-10",
                std::to_string(solved.cycles),
                tgi::stationary_status_name(solved.status)});
        }
    }
    experiment_support::progress("right-hand-side robustness");
    const auto rhs_problem = experiment_support::make_problem(
        grid, field, config);
    const auto rhs_initial = tgi::build_geometric_interpolation(grid);
    const auto finite_transfer = make_transfer(
        grid, rhs_problem.matrix, rhs_initial,
        Method::FiniteOneThird, threads);
    const auto endpoint_transfer = make_transfer(
        grid, rhs_problem.matrix, rhs_initial,
        Method::EnergyEndpoint, threads);
    const tgi::TwoGridCycle finite_cycle(
        rhs_problem.matrix, finite_transfer, 1, threads);
    const tgi::TwoGridCycle endpoint_cycle(
        rhs_problem.matrix, endpoint_transfer, 1, threads);
    experiment_support::Rows rhs_rows;
    for (const auto& rhs_case : make_rhs_cases(grid)) {
        for (const auto& item : {
                 std::pair<std::string, const tgi::TwoGridCycle*>{
                     "finite-1/3", &finite_cycle},
                 {"energy-endpoint", &endpoint_cycle}}) {
            const auto solved = tgi::solve_two_grid(
                rhs_case.values, *item.second, 1.0e-6, maximum_cycles);
            rhs_rows.push_back({
                rhs_case.name, item.first,
                item.first == "finite-1/3"
                    ? "m=" + std::to_string(
                          experiment_support::fixed_path_steps(
                              config.fine_intervals, 3))
                    : "relres<=1e-10",
                std::to_string(solved.cycles),
                tgi::stationary_status_name(solved.status)});
        }
    }
    save(seed_rows, rhs_rows, timing());
    return 0;
}
