#pragma once
#include "core/linear_algebra.hpp"
#include <exception>
#include <mutex>
namespace tgi {
class TwoGridCycle {
public:
    struct Workspace {
        Vector coarse_rhs;
        Vector coarse_solution;
        Vector coarse_work;
    };
    TwoGridCycle(const SparseMatrix& a, const SparseMatrix& p,
                 int smoothing_steps = 1, int setup_threads = 1);
    double iterate(const Vector& rhs, Vector& solution, Vector& residual,
                   Workspace& workspace) const;
    void apply(const Vector& rhs, Vector& solution, Vector& residual,
               Workspace& workspace) const;
    const SparseMatrix& coarse_matrix() const { return coarse_matrix_; }
    double interpolation_energy() const {
        double energy = 0.0;
        for (double value : coarse_matrix_.diagonal()) energy += 0.5 * value;
        return energy;
    }
private:
    void solve_gauss_seidel_sweep(const Vector& rhs, bool forward,
                                  Vector& solution) const;
    void restrict_fine_residual(const Vector& rhs, const Vector& solution,
                                Vector& coarse_rhs) const;
    const SparseMatrix& a_;
    const SparseMatrix& p_;
    Vector inverse_diagonal_;
    std::vector<int> diagonal_position_;
    int smoothing_steps_;
    int application_threads_ = 1;
    SparseMatrix p_transpose_;
    BlockedSparseMatrix restriction_operator_;
    BlockedSparseMatrix prolongation_operator_;
    SparseMatrix coarse_matrix_;
    SparseCholesky coarse_solver_;
};
enum class StationaryIterationStatus {
    Converged,
    SlowAtLimit,
    Diverged
};
inline const char* stationary_status_name(StationaryIterationStatus status) {
    if (status == StationaryIterationStatus::Converged) return "converged";
    if (status == StationaryIterationStatus::SlowAtLimit) return "slow-limit";
    return "diverged";
}
struct StationaryIterationResult {
    int cycles = 0;
    double relative_residual = 1.0;
    double effective_factor = 1.0;
    StationaryIterationStatus status = StationaryIterationStatus::SlowAtLimit;
};
namespace two_grid_solver_detail {
inline void nested_dissection_rectangle(int side, int separator_width,
                                        int xmin, int xmax,
                                        int ymin, int ymax,
                                        std::vector<int>& permutation) {
    const int width = xmax - xmin;
    const int height = ymax - ymin;
    if (width <= 0 || height <= 0) return;
    if (width <= 2 * separator_width + 4 ||
        height <= 2 * separator_width + 4) {
        for (int y = ymin; y < ymax; ++y) {
            for (int x = xmin; x < xmax; ++x) {
                permutation.push_back(y * side + x);
            }
        }
        return;
    }
    if (width >= height) {
        const int separator_begin =
            xmin + (width - separator_width) / 2;
        const int separator_end =
            separator_begin + separator_width;
        nested_dissection_rectangle(
            side, separator_width, xmin, separator_begin,
            ymin, ymax, permutation);
        nested_dissection_rectangle(
            side, separator_width, separator_end, xmax,
            ymin, ymax, permutation);
        for (int y = ymin; y < ymax; ++y) {
            for (int x = separator_begin; x < separator_end; ++x) {
                permutation.push_back(y * side + x);
            }
        }
    } else {
        const int separator_begin =
            ymin + (height - separator_width) / 2;
        const int separator_end =
            separator_begin + separator_width;
        nested_dissection_rectangle(
            side, separator_width, xmin, xmax,
            ymin, separator_begin, permutation);
        nested_dissection_rectangle(
            side, separator_width, xmin, xmax,
            separator_end, ymax, permutation);
        for (int y = separator_begin; y < separator_end; ++y) {
            for (int x = xmin; x < xmax; ++x) {
                permutation.push_back(y * side + x);
            }
        }
    }
}
inline std::vector<int> coarse_ordering(const SparseMatrix& matrix) {
    const int size = matrix.rows();
    const int side = static_cast<int>(
        std::llround(std::sqrt(static_cast<double>(size))));
    int separator_width = 1;
    for (int row = 0; row < size; ++row) {
        const int row_x = row % side;
        const int row_y = row / side;
        for (int position =
                 matrix.row_ptr()[static_cast<std::size_t>(row)];
             position <
                 matrix.row_ptr()[static_cast<std::size_t>(row) + 1U];
             ++position) {
            const int col =
                matrix.col_idx()[static_cast<std::size_t>(position)];
            separator_width = std::max(
                separator_width,
                std::max(std::abs(row_x - col % side),
                         std::abs(row_y - col / side)));
        }
    }
    std::vector<int> permutation;
    permutation.reserve(static_cast<std::size_t>(size));
    nested_dissection_rectangle(
        side, separator_width, 0, side, 0, side, permutation);
    return permutation;
}
inline SparseMatrix symmetrize_upper(const SparseMatrix& upper) {
    std::vector<int> lower_counts(
        static_cast<std::size_t>(upper.rows()), 0);
    std::vector<int> row_ptr(
        static_cast<std::size_t>(upper.rows()) + 1U, 0);
    for (int row = 0; row < upper.rows(); ++row) {
        int row_entries = 0;
        for (int position =
                 upper.row_ptr()[static_cast<std::size_t>(row)];
             position <
                 upper.row_ptr()[static_cast<std::size_t>(row) + 1U];
             ++position) {
            const int col =
                upper.col_idx()[static_cast<std::size_t>(position)];
            ++row_entries;
            if (col != row) {
                ++lower_counts[static_cast<std::size_t>(col)];
            }
        }
        row_ptr[static_cast<std::size_t>(row) + 1U] = row_entries;
    }
    for (int row = 0; row < upper.rows(); ++row) {
        row_ptr[static_cast<std::size_t>(row) + 1U] +=
            row_ptr[static_cast<std::size_t>(row)] +
            lower_counts[static_cast<std::size_t>(row)];
    }
    std::vector<int> lower_next = row_ptr;
    std::vector<int> upper_next(static_cast<std::size_t>(upper.rows()));
    for (int row = 0; row < upper.rows(); ++row) {
        upper_next[static_cast<std::size_t>(row)] =
            row_ptr[static_cast<std::size_t>(row)] +
            lower_counts[static_cast<std::size_t>(row)];
    }
    std::vector<int> col_idx(static_cast<std::size_t>(row_ptr.back()));
    Vector values(static_cast<std::size_t>(row_ptr.back()));
    for (int row = 0; row < upper.rows(); ++row) {
        for (int position =
                 upper.row_ptr()[static_cast<std::size_t>(row)];
             position <
                 upper.row_ptr()[static_cast<std::size_t>(row) + 1U];
             ++position) {
            const int col =
                upper.col_idx()[static_cast<std::size_t>(position)];
            const double value =
                upper.values()[static_cast<std::size_t>(position)];
            const int target =
                upper_next[static_cast<std::size_t>(row)]++;
            col_idx[static_cast<std::size_t>(target)] = col;
            values[static_cast<std::size_t>(target)] = value;
            if (col != row) {
                const int mirror =
                    lower_next[static_cast<std::size_t>(col)]++;
                col_idx[static_cast<std::size_t>(mirror)] = row;
                values[static_cast<std::size_t>(mirror)] = value;
            }
        }
    }
    return SparseMatrix(
        upper.rows(), upper.cols(), std::move(row_ptr),
        std::move(col_idx), std::move(values));
}
inline void gauss_seidel_sweep(
    const SparseMatrix& matrix, const Vector& inverse_diagonal,
    const std::vector<int>& diagonal_position, const Vector& rhs,
    bool forward, Vector& solution) {
    const int begin = forward ? 0 : matrix.rows() - 1;
    const int end = forward ? matrix.rows() : -1;
    const int increment = forward ? 1 : -1;
    for (int row = begin; row != end; row += increment) {
        double value = rhs[static_cast<std::size_t>(row)];
        const int diagonal =
            diagonal_position[static_cast<std::size_t>(row)];
        for (int position = matrix.row_ptr()[static_cast<std::size_t>(row)];
             position < diagonal; ++position) {
            value -= matrix.values()[static_cast<std::size_t>(position)] *
                solution[static_cast<std::size_t>(
                    matrix.col_idx()[static_cast<std::size_t>(position)])];
        }
        for (int position = diagonal + 1;
             position < matrix.row_ptr()[static_cast<std::size_t>(row) + 1U];
             ++position) {
            value -= matrix.values()[static_cast<std::size_t>(position)] *
                solution[static_cast<std::size_t>(
                    matrix.col_idx()[static_cast<std::size_t>(position)])];
        }
        solution[static_cast<std::size_t>(row)] = value *
            inverse_diagonal[static_cast<std::size_t>(row)];
    }
}
inline SparseMatrix multiply_sparse_matrices(
    const SparseMatrix& lhs, const SparseMatrix& rhs,
    int thread_count,
    bool upper_triangle_only) {
    const int worker_count = std::max(
        1, std::min(lhs.rows(), thread_count));
    const std::size_t dense_size = static_cast<std::size_t>(rhs.rows()) * static_cast<std::size_t>(rhs.cols());
    Vector dense_rhs;
    if (upper_triangle_only && rhs.cols() >= 32 && dense_size <= 16000000U && rhs.nnz() * 2U >= dense_size) {
        dense_rhs.assign(dense_size, 0.0);
#pragma omp parallel for schedule(static) num_threads(worker_count) if(worker_count > 1)
        for (int row = 0; row < rhs.rows(); ++row) {
            double* target = dense_rhs.data() + static_cast<std::size_t>(row) * static_cast<std::size_t>(rhs.cols());
            for (int pos = rhs.row_ptr()[static_cast<std::size_t>(row)]; pos < rhs.row_ptr()[static_cast<std::size_t>(row) + 1U]; ++pos) {
                target[rhs.col_idx()[static_cast<std::size_t>(pos)]] = rhs.values()[static_cast<std::size_t>(pos)];
            }
        }
    }
    struct RowBlockProduct {
        int first_row = 0;
        int last_row = 0;
        std::vector<int> row_offsets;
        std::vector<int> columns;
        std::vector<double> entries;
    };
    std::vector<RowBlockProduct> products(
        static_cast<std::size_t>(worker_count));
    std::exception_ptr worker_error;
    std::mutex error_mutex;
    auto worker = [&](int worker_id) {
        try {
            RowBlockProduct& product =
                products[static_cast<std::size_t>(worker_id)];
            product.first_row =
                lhs.rows() * worker_id / worker_count;
            product.last_row =
                lhs.rows() * (worker_id + 1) / worker_count;
            if (upper_triangle_only) {
                const auto boundary = [&](int partition) {
                    if (partition == worker_count) return lhs.rows();
                    const double fraction = static_cast<double>(partition) / static_cast<double>(worker_count);
                    const int alignment = dense_rhs.empty() ? 1 : 4;
                    return alignment * static_cast<int>(static_cast<double>(lhs.rows()) * (1.0 - std::sqrt(1.0 - fraction)) / static_cast<double>(alignment));
                };
                product.first_row = boundary(worker_id);
                product.last_row = boundary(worker_id + 1);
            }
            const int local_rows =
                product.last_row - product.first_row;
            product.row_offsets.assign(
                static_cast<std::size_t>(local_rows) + 1U, 0);
            const std::size_t reserve_entries =
                static_cast<std::size_t>(local_rows) *
                static_cast<std::size_t>(
                    std::min(rhs.cols(), 64));
            product.columns.reserve(reserve_entries);
            product.entries.reserve(reserve_entries);
            if (!dense_rhs.empty()) {
                Vector sums(static_cast<std::size_t>(rhs.cols()) * 4U);
                for (int first = product.first_row; first < product.last_row; first += 4) {
                    const int lanes = std::min(4, product.last_row - first);
                    std::fill(sums.begin(), sums.end(), 0.0);
                    int next[4]{}, end[4]{};
                    for (int lane = 0; lane < lanes; ++lane) {
                        const auto row = static_cast<std::size_t>(first + lane);
                        next[lane] = lhs.row_ptr()[row];
                        end[lane] = lhs.row_ptr()[row + 1U];
                    }
                    while (true) {
                        int inner = rhs.rows();
                        for (int lane = 0; lane < lanes; ++lane) if (next[lane] < end[lane]) inner = std::min(inner, lhs.col_idx()[static_cast<std::size_t>(next[lane])]);
                        if (inner == rhs.rows()) break;
                        const double* source = dense_rhs.data() + static_cast<std::size_t>(inner) * static_cast<std::size_t>(rhs.cols());
                        for (int lane = 0; lane < lanes; ++lane) {
                            if (next[lane] == end[lane] || lhs.col_idx()[static_cast<std::size_t>(next[lane])] != inner) continue;
                            const double value = lhs.values()[static_cast<std::size_t>(next[lane]++)];
                            double* target = sums.data() + static_cast<std::size_t>(lane) * static_cast<std::size_t>(rhs.cols());
#pragma omp simd
                            for (int col = first + lane; col < rhs.cols(); ++col) target[col] += value * source[col];
                        }
                    }
                    for (int lane = 0; lane < lanes; ++lane) {
                        const double* source = sums.data() + static_cast<std::size_t>(lane) * static_cast<std::size_t>(rhs.cols());
                        for (int col = first + lane; col < rhs.cols(); ++col) if (std::abs(source[col]) > 0.0) {
                            product.columns.push_back(col);
                            product.entries.push_back(source[col]);
                        }
                        product.row_offsets[static_cast<std::size_t>(first + lane - product.first_row) + 1U] = static_cast<int>(product.entries.size());
                    }
                }
                return;
            }
            std::vector<double> accumulator(
                static_cast<std::size_t>(rhs.cols()), 0.0);
            std::vector<int> marker(
                static_cast<std::size_t>(rhs.cols()), -1);
            std::vector<int> touched;
            touched.reserve(static_cast<std::size_t>(
                std::min(rhs.cols(), 256)));
            for (int row = product.first_row;
                 row < product.last_row; ++row) {
                if (upper_triangle_only) {
                    std::fill(
                        accumulator.begin() + row,
                        accumulator.end(), 0.0);
                    for (int lhs_position =
                             lhs.row_ptr()[static_cast<std::size_t>(row)];
                         lhs_position <
                             lhs.row_ptr()[
                                 static_cast<std::size_t>(row) + 1U];
                         ++lhs_position) {
                        const int inner =
                            lhs.col_idx()[
                                static_cast<std::size_t>(lhs_position)];
                        const double lhs_value =
                            lhs.values()[
                                static_cast<std::size_t>(lhs_position)];
                        const int rhs_row_begin =
                            rhs.row_ptr()[static_cast<std::size_t>(inner)];
                        const int rhs_end =
                            rhs.row_ptr()[
                                static_cast<std::size_t>(inner) + 1U];
                        const int rhs_begin = static_cast<int>(
                            std::lower_bound(
                                rhs.col_idx().begin() + rhs_row_begin,
                                rhs.col_idx().begin() + rhs_end,
                                row) -
                            rhs.col_idx().begin());
                        for (int rhs_position = rhs_begin;
                             rhs_position < rhs_end;
                             ++rhs_position) {
                            const int col =
                                rhs.col_idx()[
                                    static_cast<std::size_t>(rhs_position)];
                            accumulator[static_cast<std::size_t>(col)] +=
                                lhs_value *
                                rhs.values()[
                                    static_cast<std::size_t>(rhs_position)];
                        }
                    }
                    for (int col = row; col < rhs.cols(); ++col) {
                        const double value =
                            accumulator[static_cast<std::size_t>(col)];
                        if (std::abs(value) > 0.0) {
                            product.columns.push_back(col);
                            product.entries.push_back(value);
                        }
                    }
                    product.row_offsets[
                        static_cast<std::size_t>(
                            row - product.first_row) + 1U] =
                        static_cast<int>(product.entries.size());
                    continue;
                }
                touched.clear();
                for (int lhs_position =
                         lhs.row_ptr()[static_cast<std::size_t>(row)];
                     lhs_position <
                         lhs.row_ptr()[static_cast<std::size_t>(row) + 1U];
                     ++lhs_position) {
                    const int inner =
                        lhs.col_idx()[static_cast<std::size_t>(lhs_position)];
                    const double lhs_value =
                        lhs.values()[static_cast<std::size_t>(lhs_position)];
                    const int rhs_begin =
                        rhs.row_ptr()[static_cast<std::size_t>(inner)];
                    const int rhs_end =
                        rhs.row_ptr()[static_cast<std::size_t>(inner) + 1U];
                    for (int rhs_position = rhs_begin;
                         rhs_position < rhs_end;
                         ++rhs_position) {
                        const int col =
                            rhs.col_idx()[static_cast<std::size_t>(rhs_position)];
                        if (marker[static_cast<std::size_t>(col)] != row) {
                            marker[static_cast<std::size_t>(col)] = row;
                            accumulator[static_cast<std::size_t>(col)] = 0.0;
                            touched.push_back(col);
                        }
                        accumulator[static_cast<std::size_t>(col)] +=
                            lhs_value *
                            rhs.values()[static_cast<std::size_t>(rhs_position)];
                    }
                }
                const int candidate_count = rhs.cols();
                const bool dense_scan =
                    candidate_count > 0 &&
                    touched.size() * 2U >=
                        static_cast<std::size_t>(candidate_count);
                if (dense_scan) {
                    for (int col = 0; col < rhs.cols(); ++col) {
                        if (marker[static_cast<std::size_t>(col)] != row) {
                            continue;
                        }
                        const double value =
                            accumulator[static_cast<std::size_t>(col)];
                        if (std::abs(value) > 0.0) {
                            product.columns.push_back(col);
                            product.entries.push_back(value);
                        }
                    }
                } else {
                    std::sort(touched.begin(), touched.end());
                    for (int col : touched) {
                        const double value =
                            accumulator[static_cast<std::size_t>(col)];
                        if (std::abs(value) > 0.0) {
                            product.columns.push_back(col);
                            product.entries.push_back(value);
                        }
                    }
                }
                product.row_offsets[
                    static_cast<std::size_t>(
                        row - product.first_row) + 1U] =
                    static_cast<int>(product.entries.size());
            }
        } catch (...) {
            std::lock_guard<std::mutex> lock(error_mutex);
            if (!worker_error) worker_error = std::current_exception();
        }
    };
    if (worker_count == 1) {
        worker(0);
    } else {
#pragma omp parallel for schedule(static) num_threads(worker_count)
        for (int worker_id = 0;
             worker_id < worker_count; ++worker_id) {
            worker(worker_id);
        }
    }
    if (worker_error) std::rethrow_exception(worker_error);
    if (worker_count == 1) {
        RowBlockProduct& product = products.front();
        SparseMatrix result(
            lhs.rows(), rhs.cols(), std::move(product.row_offsets),
            std::move(product.columns), std::move(product.entries));
        return upper_triangle_only
            ? symmetrize_upper(result)
            : std::move(result);
    }
    std::vector<int> row_ptr(
        static_cast<std::size_t>(lhs.rows()) + 1U, 0);
    for (const RowBlockProduct& product : products) {
        for (int row = product.first_row; row < product.last_row; ++row) {
            const int local_row = row - product.first_row;
            const int count =
                product.row_offsets[
                    static_cast<std::size_t>(local_row) + 1U] -
                product.row_offsets[static_cast<std::size_t>(local_row)];
            row_ptr[static_cast<std::size_t>(row) + 1U] =
                row_ptr[static_cast<std::size_t>(row)] + count;
        }
    }
    std::vector<int> col_idx(
        static_cast<std::size_t>(row_ptr.back()));
    std::vector<double> values(
        static_cast<std::size_t>(row_ptr.back()));
    for (RowBlockProduct& product : products) {
        const std::size_t begin = static_cast<std::size_t>(
            row_ptr[static_cast<std::size_t>(product.first_row)]);
        std::move(product.columns.begin(), product.columns.end(),
                  col_idx.begin() + static_cast<std::ptrdiff_t>(begin));
        std::move(product.entries.begin(), product.entries.end(),
                  values.begin() + static_cast<std::ptrdiff_t>(begin));
    }
    SparseMatrix result(lhs.rows(), rhs.cols(), std::move(row_ptr),
                        std::move(col_idx), std::move(values));
    return upper_triangle_only
        ? symmetrize_upper(result)
        : std::move(result);
}
}
inline TwoGridCycle::TwoGridCycle(const SparseMatrix& a, const SparseMatrix& p,
                                  int smoothing_steps, int setup_threads)
    : a_(a), p_(p), smoothing_steps_(smoothing_steps) {
    inverse_diagonal_.resize(static_cast<std::size_t>(a_.rows()));
    diagonal_position_.resize(static_cast<std::size_t>(a_.rows()));
    for (int row = 0; row < a_.rows(); ++row) {
        const std::size_t i = static_cast<std::size_t>(row);
        int position = a_.row_ptr()[i];
        const int end = a_.row_ptr()[i + 1U];
        while (position < end &&
               a_.col_idx()[static_cast<std::size_t>(position)] < row) {
            ++position;
        }
        const double diagonal =
            a_.values()[static_cast<std::size_t>(position)];
        inverse_diagonal_[i] = 1.0 / diagonal;
        diagonal_position_[i] = position;
    }
    p_transpose_ = p_.transpose(setup_threads);
    const SparseMatrix ap = two_grid_solver_detail::multiply_sparse_matrices(
        a_, p_, setup_threads, false);
    coarse_matrix_ = two_grid_solver_detail::multiply_sparse_matrices(
        p_transpose_, ap, setup_threads, true);
    coarse_solver_.factorize(
        coarse_matrix_, two_grid_solver_detail::coarse_ordering(coarse_matrix_));
    constexpr std::size_t parallel_threshold = 40000U;
    application_threads_ = p_.nnz() >= parallel_threshold
        ? std::max(
              1, std::min(
                     {8, setup_threads,
                      p_.rows(), p_.cols()}))
        : 1;
    prolongation_operator_ = BlockedSparseMatrix(p_);
    restriction_operator_ = BlockedSparseMatrix(p_transpose_);
    if (restriction_operator_.enabled()) p_transpose_ = SparseMatrix();
}
inline void TwoGridCycle::solve_gauss_seidel_sweep(
    const Vector& rhs, bool forward, Vector& solution) const {
    two_grid_solver_detail::gauss_seidel_sweep(
        a_, inverse_diagonal_, diagonal_position_, rhs, forward, solution);
}
inline void TwoGridCycle::restrict_fine_residual(
    const Vector& rhs, const Vector& solution, Vector& coarse_rhs) const {
    coarse_rhs.assign(static_cast<std::size_t>(p_.cols()), 0.0);
    for (int row = 0; row < a_.rows(); ++row) {
        double product = 0.0;
        for (int position =
                 a_.row_ptr()[static_cast<std::size_t>(row)];
             position <
                 a_.row_ptr()[static_cast<std::size_t>(row) + 1U];
             ++position) {
            product += a_.values()[static_cast<std::size_t>(position)] *
                       solution[static_cast<std::size_t>(
                           a_.col_idx()[static_cast<std::size_t>(position)])];
        }
        const double residual =
            rhs[static_cast<std::size_t>(row)] - product;
        for (int position =
                 p_.row_ptr()[static_cast<std::size_t>(row)];
             position <
                 p_.row_ptr()[static_cast<std::size_t>(row) + 1U];
             ++position) {
            coarse_rhs[static_cast<std::size_t>(
                p_.col_idx()[static_cast<std::size_t>(position)])] +=
                p_.values()[static_cast<std::size_t>(position)] * residual;
        }
    }
}
inline void TwoGridCycle::apply(
    const Vector& rhs, Vector& solution, Vector& residual,
    Workspace& workspace) const {
    for (int step = 0; step < smoothing_steps_; ++step) {
        solve_gauss_seidel_sweep(rhs, true, solution);
    }
    if (application_threads_ == 1) {
        restrict_fine_residual(rhs, solution, workspace.coarse_rhs);
    } else {
        a_.residual(
            solution, rhs, residual, application_threads_);
        if (restriction_operator_.enabled()) {
            restriction_operator_.multiply(
                residual, workspace.coarse_rhs, application_threads_);
        } else {
            p_transpose_.multiply(residual, workspace.coarse_rhs,
                                  application_threads_);
        }
    }
    coarse_solver_.solve(
        workspace.coarse_rhs, workspace.coarse_solution,
        workspace.coarse_work);
    if (prolongation_operator_.enabled()) {
        prolongation_operator_.multiply(workspace.coarse_solution, solution,
                                        application_threads_, true);
    } else {
        p_.multiply_add(workspace.coarse_solution, solution,
                        application_threads_);
    }
    for (int step = 0; step < smoothing_steps_; ++step) {
        solve_gauss_seidel_sweep(rhs, false, solution);
    }
}
inline double TwoGridCycle::iterate(
    const Vector& rhs, Vector& solution, Vector& residual,
    Workspace& workspace) const {
    apply(rhs, solution, residual, workspace);
    return a_.residual_squared(solution, rhs, residual, application_threads_);
}
inline StationaryIterationResult solve_two_grid(
    const Vector& rhs, const TwoGridCycle& cycle, double relative_tolerance = 1e-8,
    int max_cycles = 40000) {
    StationaryIterationResult result;
    Vector solution(rhs.size(), 0.0);
    Vector residual;
    TwoGridCycle::Workspace workspace;
    const double initial_norm = norm2(rhs);
    if (initial_norm == 0.0) {
        result.relative_residual = 0.0;
        result.effective_factor = 0.0;
        result.status = StationaryIterationStatus::Converged;
        return result;
    }
    for (int iteration = 0; iteration < max_cycles; ++iteration) {
        const double residual_squared = cycle.iterate(rhs, solution, residual, workspace);
        result.cycles = iteration + 1;
        result.relative_residual = std::sqrt(residual_squared) / initial_norm;
        if (!std::isfinite(result.relative_residual)) {
            result.status = StationaryIterationStatus::Diverged;
            break;
        }
        if (result.relative_residual <= relative_tolerance) {
            result.status = StationaryIterationStatus::Converged;
            break;
        }
    }
    if (result.cycles > 0) result.effective_factor = std::exp(
        std::log(result.relative_residual) / static_cast<double>(result.cycles));
    return result;
}
}
