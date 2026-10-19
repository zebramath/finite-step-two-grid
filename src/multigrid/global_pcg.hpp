#pragma once
#include "multigrid/energy_interpolation.hpp"
#include <exception>
#include <limits>
#include <mutex>
#include <omp.h>
namespace tgi {
class GlobalEnergyPcgPath {
public:
    GlobalEnergyPcgPath(
        const StructuredGrid& grid, const SparseMatrix& a,
        const SparseMatrix& initial_prolongation, int thread_count = 1);
    void advance_to(int target_steps);
    void advance_until_relative_residual(
        double tolerance, int maximum_steps = 40000);
    SparseMatrix prolongation() const;
private:
    struct ColumnState {
        Vector solution;
        Vector residual;
        Vector direction;
        double rz = 0.0;
        double residual_squared = 0.0;
        double rhs_norm = 1.0;
        std::vector<int> active_rows;
        std::vector<int> frontier;
        std::vector<int> next_frontier;
        std::vector<unsigned char> reached;
        bool full_domain = false;
        std::vector<int> initial_rows;
        int iterations = 0;
        bool active = true;
        void use_full_domain() {
            full_domain = true;
            std::vector<int>().swap(active_rows);
            std::vector<int>().swap(frontier);
            std::vector<int>().swap(next_frontier);
            std::vector<unsigned char>().swap(reached);
            std::vector<int>().swap(initial_rows);
        }
    };
    void expand_frontier(ColumnState& state) const;
    bool advance_one_iteration(ColumnState& state, Vector& product) const;
    void advance_columns(int maximum_steps, double tolerance);
    template <class Function>
    void for_active_rows(const ColumnState& state, Function&& function) const {
        if (state.full_domain) {
            for (std::size_t row = 0; row < state.solution.size(); ++row) {
                function(row);
            }
        } else {
            for (int row : state.active_rows) {
                function(static_cast<std::size_t>(row));
            }
        }
    }
    template <class Function>
    void for_solution_rows(const ColumnState& state, Function&& function) const {
        if (state.full_domain || state.active_rows.size() > state.solution.size() / 2U) {
            for (std::size_t row = 0; row < state.solution.size(); ++row) function(row);
            return;
        }
        std::size_t active = 0, initial = 0;
        while (active < state.active_rows.size() || initial < state.initial_rows.size()) {
            const int active_row = active < state.active_rows.size()
                ? state.active_rows[active] : system_.matrix.rows();
            const int initial_row = initial < state.initial_rows.size()
                ? state.initial_rows[initial] : system_.matrix.rows();
            const int row = std::min(active_row, initial_row);
            if (active_row == row) ++active;
            if (initial_row == row) ++initial;
            function(static_cast<std::size_t>(row));
        }
    }
    const StructuredGrid& grid_;
    energy_interpolation_detail::GlobalFSystem system_;
    std::vector<ColumnState> columns_;
    std::vector<Vector> products_;
    int thread_count_ = 1;
};
inline GlobalEnergyPcgPath::GlobalEnergyPcgPath(
    const StructuredGrid& grid, const SparseMatrix& a,
    const SparseMatrix& initial_prolongation, int thread_count)
    : grid_(grid) {
    system_ = energy_interpolation_detail::assemble_global_f_system(grid, a);
    thread_count_ = std::max(1, std::min(thread_count, grid.coarse_size()));
    columns_.resize(static_cast<std::size_t>(grid.coarse_size()));
    products_.resize(static_cast<std::size_t>(thread_count_));
    for (Vector& product : products_) product.resize(system_.f_nodes.size());
    const SparseMatrix initial_transpose =
        initial_prolongation.transpose(thread_count_);
    std::vector<unsigned char> candidate_mask(system_.f_nodes.size(), 0);
    std::vector<int> candidates;
    for (int coarse = 0; coarse < grid.coarse_size(); ++coarse) {
        ColumnState& state = columns_[static_cast<std::size_t>(coarse)];
        const std::size_t n = system_.f_nodes.size();
        state.solution.assign(n, 0.0);
        for (int position =
                 initial_transpose.row_ptr()[static_cast<std::size_t>(coarse)];
             position < initial_transpose.row_ptr()[
                 static_cast<std::size_t>(coarse) + 1U]; ++position) {
            const int fine = initial_transpose.col_idx()[
                static_cast<std::size_t>(position)];
            if (grid.is_coarse_node(fine)) continue;
            const int local =
                system_.local_index[static_cast<std::size_t>(fine)];
            state.solution[static_cast<std::size_t>(local)] =
                initial_transpose.values()[static_cast<std::size_t>(position)];
            state.initial_rows.push_back(local);
        }
        state.residual.assign(n, 0.0);
        double rhs_squared = 0.0;
        for (const auto& [row, value] :
             system_.rhs_entries[static_cast<std::size_t>(coarse)]) {
            state.residual[static_cast<std::size_t>(row)] += value;
            rhs_squared += value * value;
        }
        candidates.clear();
        const auto add_candidate = [&](int row) {
            const auto index = static_cast<std::size_t>(row);
            if (candidate_mask[index] == 0) {
                candidate_mask[index] = 1;
                candidates.push_back(row);
            }
        };
        for (int row : state.initial_rows) {
            for (int pos = system_.matrix.row_ptr()[static_cast<std::size_t>(row)];
                 pos < system_.matrix.row_ptr()[static_cast<std::size_t>(row) + 1U]; ++pos) {
                add_candidate(system_.matrix.col_idx()[static_cast<std::size_t>(pos)]);
            }
        }
        for (const auto& entry : system_.rhs_entries[static_cast<std::size_t>(coarse)]) {
            add_candidate(entry.first);
        }
        std::sort(candidates.begin(), candidates.end());
        for (int row : candidates) {
            const auto index = static_cast<std::size_t>(row);
            double product = 0.0;
            for (int pos = system_.matrix.row_ptr()[index];
                 pos < system_.matrix.row_ptr()[index + 1U]; ++pos) {
                product += system_.matrix.values()[static_cast<std::size_t>(pos)] *
                    state.solution[static_cast<std::size_t>(system_.matrix.col_idx()[static_cast<std::size_t>(pos)])];
            }
            state.residual[index] += -product;
            if (state.residual[index] != 0.0) state.active_rows.push_back(row);
            candidate_mask[index] = 0;
        }
        const double residual_scale = std::max(std::sqrt(rhs_squared), 1.0e-30);
        state.rhs_norm = residual_scale;
        state.frontier = state.active_rows;
        state.reached.assign(n, 0);
        for (int row : state.active_rows) {
            state.reached[static_cast<std::size_t>(row)] = 1;
        }
        state.direction.resize(n);
        state.rz = 0.0;
        state.residual_squared = 0.0;
        for (int row : state.active_rows) {
            const auto index = static_cast<std::size_t>(row);
            state.direction[index] = system_.inverse_diagonal[index] *
                state.residual[index];
            state.rz += state.residual[index] * state.direction[index];
            state.residual_squared +=
                state.residual[index] * state.residual[index];
        }
        const double threshold = std::numeric_limits<double>::epsilon() *
            residual_scale;
        state.active = std::isfinite(state.rz) && state.rz > 0.0 &&
            state.residual_squared > threshold * threshold;
    }
}
inline void GlobalEnergyPcgPath::expand_frontier(ColumnState& state) const {
    if (state.full_domain) return;
    auto& next = state.next_frontier;
    next.clear();
    next.reserve(state.frontier.size() * 2U);
    const auto& offsets = system_.matrix.row_ptr();
    const auto& columns = system_.matrix.col_idx();
    for (int row : state.frontier) {
        for (int pos = offsets[static_cast<std::size_t>(row)];
             pos < offsets[static_cast<std::size_t>(row) + 1U]; ++pos) {
            const int column = columns[static_cast<std::size_t>(pos)];
            if (state.reached[static_cast<std::size_t>(column)] == 0) {
                state.reached[static_cast<std::size_t>(column)] = 1;
                next.push_back(column);
            }
        }
    }
    if (state.active_rows.size() + next.size() == state.solution.size()) {
        state.use_full_domain();
        return;
    }
    std::sort(next.begin(), next.end());
    const auto old_size = static_cast<std::ptrdiff_t>(state.active_rows.size());
    state.active_rows.insert(state.active_rows.end(), next.begin(), next.end());
    std::inplace_merge(state.active_rows.begin(),
                      state.active_rows.begin() + old_size,
                      state.active_rows.end());
    state.frontier.swap(next);
}
inline bool GlobalEnergyPcgPath::advance_one_iteration(
    ColumnState& state, Vector& product) const {
    expand_frontier(state);
    double denominator = 0.0;
    for_active_rows(state, [&](std::size_t row) {
        const double sum = system_.row_product(row, state.direction);
        product[row] = sum;
        denominator += state.direction[row] * sum;
    });
    if (!(denominator > 0.0) || !std::isfinite(denominator) ||
        !(state.rz > 0.0) || !std::isfinite(state.rz)) {
        throw std::runtime_error("global PCG path lost positive curvature");
    }
    const double alpha = state.rz / denominator;
    double rz_new = 0.0;
    state.residual_squared = 0.0;
    for_active_rows(state, [&](std::size_t row) {
        state.solution[row] += alpha * state.direction[row];
        state.residual[row] += -alpha * product[row];
        const double z = system_.inverse_diagonal[row] * state.residual[row];
        rz_new += state.residual[row] * z;
        state.residual_squared += state.residual[row] * state.residual[row];
    });
    ++state.iterations;
    if (!(rz_new > 0.0) || !std::isfinite(rz_new)) {
        state.rz = rz_new;
        state.active = false;
        const double threshold = std::numeric_limits<double>::epsilon() *
            state.rhs_norm;
        if (!std::isfinite(rz_new) ||
            state.residual_squared > threshold * threshold) {
            throw std::runtime_error("global PCG path broke down before convergence");
        }
        return false;
    }
    const double beta = rz_new / state.rz;
    for_active_rows(state, [&](std::size_t row) {
        state.direction[row] = system_.inverse_diagonal[row] *
            state.residual[row] + beta * state.direction[row];
    });
    state.rz = rz_new;
    return true;
}
inline void GlobalEnergyPcgPath::advance_columns(
    int maximum_steps, double tolerance) {
    std::exception_ptr worker_error;
    std::mutex error_mutex;
    const auto advance = [&](int coarse, Vector& product) {
        try {
            ColumnState& state = columns_[static_cast<std::size_t>(coarse)];
            const double target = tolerance * state.rhs_norm;
            while (state.active && state.iterations < maximum_steps &&
                   state.residual_squared > target * target) {
                if (!advance_one_iteration(state, product)) break;
            }
        } catch (...) {
            std::lock_guard<std::mutex> lock(error_mutex);
            if (!worker_error) worker_error = std::current_exception();
        }
    };
#pragma omp parallel num_threads(thread_count_)
    {
        Vector& product = products_[static_cast<std::size_t>(omp_get_thread_num())];
#pragma omp for schedule(dynamic)
        for (int coarse = 0; coarse < grid_.coarse_size(); ++coarse) {
            advance(coarse, product);
        }
    }
    if (worker_error) std::rethrow_exception(worker_error);
}
inline void GlobalEnergyPcgPath::advance_to(int target_steps) {
    advance_columns(target_steps, 0.0);
}
inline void GlobalEnergyPcgPath::advance_until_relative_residual(
    double tolerance, int maximum_steps) {
    std::vector<Vector> absolute_actions(static_cast<std::size_t>(thread_count_), Vector(system_.f_nodes.size()));
    for (int replacement = 0; replacement < 16; ++replacement) {
        advance_columns(maximum_steps, tolerance);
        bool verified = true;
#pragma omp parallel for schedule(static) num_threads(thread_count_) reduction(&&:verified) if(thread_count_ > 1)
        for (std::size_t coarse = 0; coarse < columns_.size(); ++coarse) {
            Vector& residual = products_[static_cast<std::size_t>(omp_get_thread_num())];
            Vector& absolute_action = absolute_actions[static_cast<std::size_t>(omp_get_thread_num())];
            ColumnState& state = columns_[coarse];
            for (std::size_t row = 0; row < residual.size(); ++row) {
                double sum = 0.0, absolute_sum = 0.0;
                for (int pos = system_.matrix.row_ptr()[row];
                     pos < system_.matrix.row_ptr()[row + 1U]; ++pos) {
                    const double term = system_.matrix.values()[
                        static_cast<std::size_t>(pos)] * state.solution[
                        static_cast<std::size_t>(system_.matrix.col_idx()[
                            static_cast<std::size_t>(pos)])];
                    sum += term;
                    absolute_sum += std::abs(term);
                }
                residual[row] = -sum;
                absolute_action[row] = absolute_sum;
            }
            for (const auto& [row, value] : system_.rhs_entries[coarse]) {
                residual[static_cast<std::size_t>(row)] += value;
                absolute_action[static_cast<std::size_t>(row)] += std::abs(value);
            }
            const double rounding = 16.0 * std::numeric_limits<double>::epsilon() *
                norm2(absolute_action);
            if (norm2(residual) > tolerance * state.rhs_norm + rounding) {
                verified = false;
                state.residual = residual;
                state.rz = 0.0;
                state.residual_squared = dot(residual, residual);
                for (std::size_t row = 0; row < residual.size(); ++row) {
                    state.direction[row] = system_.inverse_diagonal[row] * residual[row];
                    state.rz += residual[row] * state.direction[row];
                }
                state.active = state.rz > 0.0;
                state.use_full_domain();
            }
        }
        if (verified) return;
    }
    throw std::runtime_error("endpoint true residual could not be verified");
}
inline SparseMatrix GlobalEnergyPcgPath::prolongation() const {
    std::vector<int> row_ptr(
        static_cast<std::size_t>(grid_.fine_size()) + 1U, 0);
    for (int coarse = 0; coarse < grid_.coarse_size(); ++coarse) {
        ++row_ptr[static_cast<std::size_t>(
            grid_.coarse_fine_id(coarse)) + 1U];
        const Vector& weights =
            columns_[static_cast<std::size_t>(coarse)].solution;
        for_solution_rows(columns_[static_cast<std::size_t>(coarse)], [&](std::size_t local) {
            if (weights[local] != 0.0) ++row_ptr[static_cast<std::size_t>(system_.f_nodes[local]) + 1U];
        });
    }
    for (int row = 0; row < grid_.fine_size(); ++row) {
        row_ptr[static_cast<std::size_t>(row) + 1U] +=
            row_ptr[static_cast<std::size_t>(row)];
    }
    std::vector<int> next = row_ptr;
    std::vector<int> col_idx(static_cast<std::size_t>(row_ptr.back()));
    Vector values(static_cast<std::size_t>(row_ptr.back()));
    for (int coarse = 0; coarse < grid_.coarse_size(); ++coarse) {
        const int injection_row = grid_.coarse_fine_id(coarse);
        int target = next[static_cast<std::size_t>(injection_row)]++;
        col_idx[static_cast<std::size_t>(target)] = coarse;
        values[static_cast<std::size_t>(target)] = 1.0;
        const Vector& weights =
            columns_[static_cast<std::size_t>(coarse)].solution;
        for_solution_rows(columns_[static_cast<std::size_t>(coarse)], [&](std::size_t local) {
            if (weights[local] == 0.0) return;
            target = next[static_cast<std::size_t>(system_.f_nodes[local])]++;
            col_idx[static_cast<std::size_t>(target)] = coarse;
            values[static_cast<std::size_t>(target)] = weights[local];
        });
    }
    return SparseMatrix(
        grid_.fine_size(), grid_.coarse_size(), std::move(row_ptr),
        std::move(col_idx), std::move(values));
}
}
