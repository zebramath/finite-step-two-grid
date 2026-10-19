#include "experiment/problem.hpp"
#include "experiment/reporting.hpp"
#include "multigrid/global_pcg.hpp"
namespace {
void write_sparse(
    std::ofstream& stream, const std::string& kind, int step,
    const tgi::SparseMatrix& matrix) {
    for (int row = 0; row < matrix.rows(); ++row) {
        for (int position =
                 matrix.row_ptr()[static_cast<std::size_t>(row)];
             position <
                 matrix.row_ptr()[static_cast<std::size_t>(row) + 1U];
             ++position) {
            stream << kind << ',' << step << ',' << row << ','
                   << matrix.col_idx()[static_cast<std::size_t>(position)]
                   << ','
                   << matrix.values()[static_cast<std::size_t>(position)]
                   << '\n';
        }
    }
}
}
int main() {
    constexpr int maximum_steps = 56;
    const int threads = experiment_support::thread_count();
    experiment_support::BasicConfig config;
    config.fine_intervals = 16;
    config.coarse_intervals = 4;
    const auto& field = experiment_support::channel_topologies().front();
    const auto grid = experiment_support::make_grid(config);
    const auto problem = experiment_support::make_problem(
        grid, field, config);
    const auto geometric = tgi::build_geometric_interpolation(grid);
    const auto output_path = experiment_support::results_directory() /
        "experiment4.csv";
    std::ofstream stream(output_path);
    stream << std::setprecision(17);
    stream << "kind,m,row,col,value\n";
    write_sparse(stream, "A", -2, problem.matrix);
    tgi::GlobalEnergyPcgPath path(
        grid, problem.matrix, geometric, threads);
    write_sparse(stream, "P", 0, geometric);
    for (int step = 1; step <= maximum_steps; ++step) {
        experiment_support::progress(
            "local diagnostic export " + std::to_string(step) + "/" +
            std::to_string(maximum_steps));
        path.advance_to(step);
        write_sparse(stream, "P", step, path.prolongation());
    }
    path.advance_to(80);
    path.advance_until_relative_residual(1.0e-12, 40000);
    write_sparse(stream, "P_endpoint", -1, path.prolongation());
    if (!stream) {
        throw std::runtime_error(
            "failed while writing local diagnostic matrices");
    }
    return 0;
}
