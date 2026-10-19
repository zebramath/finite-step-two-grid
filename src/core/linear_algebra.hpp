#pragma once
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>
namespace tgi {
using Vector = std::vector<double>;
inline double dot(const Vector& x, const Vector& y) {
    double sum = 0.0;
    for (std::size_t i = 0; i < x.size(); ++i) sum += x[i] * y[i];
    return sum;
}
inline double norm2(const Vector& x) { return std::sqrt(dot(x, x)); }
inline void axpy(double alpha, const Vector& x, Vector& y) {
    for (std::size_t i = 0; i < x.size(); ++i) y[i] += alpha * x[i];
}
class SparseMatrix {
public:
    SparseMatrix() = default;
    SparseMatrix(int rows, int cols, std::vector<int> row_ptr,
                 std::vector<int> col_idx, std::vector<double> values);
    int rows() const { return rows_; }
    int cols() const { return cols_; }
    std::size_t nnz() const { return values_.size(); }
    const std::vector<int>& row_ptr() const { return row_ptr_; }
    const std::vector<int>& col_idx() const { return col_idx_; }
    const std::vector<double>& values() const { return values_; }
    void multiply(const Vector& x, Vector& result,
                  int thread_count = 1) const;
    void multiply_add(const Vector& x, Vector& result,
                      int thread_count = 1) const;
    void residual(const Vector& x, const Vector& rhs, Vector& result,
                  int thread_count = 1) const;
    double residual_squared(const Vector& x, const Vector& rhs,
                            Vector& residual,
                            int thread_count = 1) const;
    SparseMatrix transpose(int thread_count = 1) const;
    Vector diagonal() const;
private:
    template <bool MeasureNorm>
    double compute_residual(const Vector& x, const Vector& rhs,
                            Vector& result, int thread_count) const;
    int rows_ = 0;
    int cols_ = 0;
    std::vector<int> row_ptr_;
    std::vector<int> col_idx_;
    std::vector<double> values_;
};
class BlockedSparseMatrix {
public:
    BlockedSparseMatrix() = default;
    explicit BlockedSparseMatrix(const SparseMatrix& matrix) {
        if (matrix.nnz() < 40000U) return;
        rows_ = matrix.rows();
        const std::size_t size = static_cast<std::size_t>(rows_) * static_cast<std::size_t>(matrix.cols());
        if (size <= 16000000U && matrix.nnz() * 4U >= size * 3U) {
            dense_columns_ = matrix.cols();
            values_.assign(static_cast<std::size_t>((rows_ + 3) / 4) * static_cast<std::size_t>(dense_columns_) * 4U, 0.0);
            for (int row = 0; row < rows_; ++row) {
                const std::size_t base = static_cast<std::size_t>(row / 4) * static_cast<std::size_t>(dense_columns_) * 4U + static_cast<std::size_t>(row % 4);
                for (int pos = matrix.row_ptr()[static_cast<std::size_t>(row)]; pos < matrix.row_ptr()[static_cast<std::size_t>(row) + 1U]; ++pos) values_[base + static_cast<std::size_t>(matrix.col_idx()[static_cast<std::size_t>(pos)]) * 4U] = matrix.values()[static_cast<std::size_t>(pos)];
            }
            return;
        }
        offsets_.push_back(0);
        const auto& ptr = matrix.row_ptr();
        const auto& columns = matrix.col_idx();
        const auto& values = matrix.values();
        for (int first = 0; first < rows_; first += 4) {
            int next[4]{};
            int end[4]{};
            for (int lane = 0; lane < 4; ++lane) {
                if (first + lane < rows_) {
                    next[lane] = ptr[static_cast<std::size_t>(first + lane)];
                    end[lane] = ptr[static_cast<std::size_t>(first + lane) + 1U];
                }
            }
            while (true) {
                int column = matrix.cols();
                for (int lane = 0; lane < 4; ++lane) {
                    if (next[lane] < end[lane]) {
                        column = std::min(column,
                            columns[static_cast<std::size_t>(next[lane])]);
                    }
                }
                if (column == matrix.cols()) break;
                columns_.push_back(column);
                for (int lane = 0; lane < 4; ++lane) {
                    double value = 0.0;
                    if (next[lane] < end[lane] &&
                        columns[static_cast<std::size_t>(next[lane])] == column) {
                        value = values[static_cast<std::size_t>(next[lane]++)];
                    }
                    values_.push_back(value);
                }
            }
            offsets_.push_back(static_cast<int>(columns_.size()));
        }
        const double packed_bytes = static_cast<double>(columns_.size()) * 36.0;
        const double csr_bytes = static_cast<double>(matrix.nnz()) * 12.0;
        if (packed_bytes > 1.10 * csr_bytes) *this = BlockedSparseMatrix();
    }
    bool enabled() const { return dense_columns_ > 0 || !offsets_.empty(); }
    void multiply(const Vector& x, Vector& result, int threads,
                  bool add = false) const {
        if (!add) result.resize(static_cast<std::size_t>(rows_));
        const int blocks = (rows_ + 3) / 4;
#pragma omp parallel for schedule(static) num_threads(threads) if(threads > 1)
        for (int block = 0; block < blocks; ++block) {
            double sums[4]{};
            if (dense_columns_ > 0) {
                const double* values = values_.data() + static_cast<std::size_t>(block) * static_cast<std::size_t>(dense_columns_) * 4U;
                for (int col = 0; col < dense_columns_; ++col) {
                    const double value = x[static_cast<std::size_t>(col)];
                    const std::size_t start = static_cast<std::size_t>(col) * 4U;
                    for (std::size_t lane = 0; lane < 4U; ++lane) sums[lane] += values[start + lane] * value;
                }
            } else for (int pos = offsets_[static_cast<std::size_t>(block)];
                 pos < offsets_[static_cast<std::size_t>(block) + 1U]; ++pos) {
                const double value = x[static_cast<std::size_t>(
                    columns_[static_cast<std::size_t>(pos)])];
                const std::size_t start = static_cast<std::size_t>(pos) * 4U;
                for (std::size_t lane = 0; lane < 4U; ++lane) {
                    sums[lane] += values_[start + lane] * value;
                }
            }
            for (int lane = 0; lane < 4 && block * 4 + lane < rows_; ++lane) {
                const std::size_t row = static_cast<std::size_t>(block * 4 + lane);
                if (add) result[row] += sums[lane];
                else result[row] = sums[lane];
            }
        }
    }
private:
    int rows_ = 0;
    int dense_columns_ = 0;
    std::vector<int> offsets_;
    std::vector<int> columns_;
    Vector values_;
};
class SparseCholesky {
public:
    SparseCholesky() = default;
    void factorize(
        const SparseMatrix& matrix,
        const std::vector<int>& new_to_old_permutation);
    void solve(const Vector& rhs, Vector& result, Vector& work) const;
private:
    int n_ = 0;
    std::vector<int> new_to_old_;
    std::vector<int> column_ptr_;
    std::vector<int> row_idx_;
    std::vector<double> values_;
};
inline SparseMatrix::SparseMatrix(
    int rows, int cols, std::vector<int> row_ptr,
    std::vector<int> col_idx, std::vector<double> values)
    : rows_(rows), cols_(cols), row_ptr_(std::move(row_ptr)),
      col_idx_(std::move(col_idx)), values_(std::move(values)) {}
inline void SparseMatrix::multiply(const Vector& x, Vector& result,
                                   int thread_count) const {
    result.resize(static_cast<std::size_t>(rows_));
    const auto multiply_row = [&](int row) {
        double sum = 0.0;
        const int begin = row_ptr_[static_cast<std::size_t>(row)];
        const int end = row_ptr_[static_cast<std::size_t>(row) + 1U];
        for (int pos = begin; pos < end; ++pos) {
            sum += values_[static_cast<std::size_t>(pos)] *
                   x[static_cast<std::size_t>(
                       col_idx_[static_cast<std::size_t>(pos)])];
        }
        result[static_cast<std::size_t>(row)] = sum;
    };
    if (thread_count > 1) {
#pragma omp parallel for schedule(static) num_threads(thread_count)
        for (int row = 0; row < rows_; ++row) multiply_row(row);
        return;
    }
    for (int row = 0; row < rows_; ++row) {
        multiply_row(row);
    }
}
inline void SparseMatrix::multiply_add(const Vector& x,
                                       Vector& result, int thread_count) const {
    const auto multiply_add_row = [&](int row) {
        double sum = 0.0;
        const int begin = row_ptr_[static_cast<std::size_t>(row)];
        const int end = row_ptr_[static_cast<std::size_t>(row) + 1U];
        for (int pos = begin; pos < end; ++pos) {
            sum += values_[static_cast<std::size_t>(pos)] *
                   x[static_cast<std::size_t>(
                       col_idx_[static_cast<std::size_t>(pos)])];
        }
        result[static_cast<std::size_t>(row)] += sum;
    };
    if (thread_count > 1) {
#pragma omp parallel for schedule(static) num_threads(thread_count)
        for (int row = 0; row < rows_; ++row) multiply_add_row(row);
        return;
    }
    for (int row = 0; row < rows_; ++row) {
        multiply_add_row(row);
    }
}
template <bool MeasureNorm>
inline double SparseMatrix::compute_residual(
    const Vector& x, const Vector& rhs, Vector& result,
    int thread_count) const {
    result.resize(static_cast<std::size_t>(rows_));
    const auto residual_row = [&](int row) {
        double product = 0.0;
        for (int pos = row_ptr_[static_cast<std::size_t>(row)];
             pos < row_ptr_[static_cast<std::size_t>(row) + 1U]; ++pos) {
            product += values_[static_cast<std::size_t>(pos)] *
                x[static_cast<std::size_t>(col_idx_[static_cast<std::size_t>(pos)])];
        }
        const double value = rhs[static_cast<std::size_t>(row)] - product;
        result[static_cast<std::size_t>(row)] = value;
        if constexpr (MeasureNorm) return value * value;
        else return 0.0;
    };
    double squared_norm = 0.0;
    if (thread_count > 1) {
#pragma omp parallel for schedule(static) num_threads(thread_count) reduction(+:squared_norm)
        for (int row = 0; row < rows_; ++row) squared_norm += residual_row(row);
    } else {
        for (int row = 0; row < rows_; ++row) squared_norm += residual_row(row);
    }
    return squared_norm;
}
inline void SparseMatrix::residual(
    const Vector& x, const Vector& rhs, Vector& result,
    int thread_count) const {
    compute_residual<false>(x, rhs, result, thread_count);
}
inline double SparseMatrix::residual_squared(
    const Vector& x, const Vector& rhs, Vector& result,
    int thread_count) const {
    return compute_residual<true>(x, rhs, result, thread_count);
}
inline SparseMatrix SparseMatrix::transpose(int thread_count) const {
    if (thread_count > 1 && values_.size() >= 50000U) {
        const int workers = std::max(1, std::min(thread_count, rows_));
        const std::size_t count_size =
            static_cast<std::size_t>(workers) *
            static_cast<std::size_t>(cols_);
        std::vector<int> local_counts(count_size, 0);
#pragma omp parallel for schedule(static) num_threads(workers)
        for (int worker = 0; worker < workers; ++worker) {
            const int first = rows_ * worker / workers;
            const int last = rows_ * (worker + 1) / workers;
            int* counts =
                local_counts.data() +
                static_cast<std::ptrdiff_t>(worker) * cols_;
            for (int row = first; row < last; ++row) {
                const int begin = row_ptr_[static_cast<std::size_t>(row)];
                const int end = row_ptr_[static_cast<std::size_t>(row) + 1U];
                for (int position = begin; position < end; ++position) {
                    ++counts[col_idx_[static_cast<std::size_t>(position)]];
                }
            }
        }
        std::vector<int> transpose_row_ptr(
            static_cast<std::size_t>(cols_) + 1U, 0);
        for (int col = 0; col < cols_; ++col) {
            int count = 0;
            for (int worker = 0; worker < workers; ++worker) {
                count += local_counts[
                    static_cast<std::size_t>(worker) *
                        static_cast<std::size_t>(cols_) +
                    static_cast<std::size_t>(col)];
            }
            transpose_row_ptr[static_cast<std::size_t>(col) + 1U] =
                transpose_row_ptr[static_cast<std::size_t>(col)] + count;
        }
        std::vector<int> worker_next(count_size, 0);
        for (int col = 0; col < cols_; ++col) {
            int next = transpose_row_ptr[static_cast<std::size_t>(col)];
            for (int worker = 0; worker < workers; ++worker) {
                const std::size_t offset =
                    static_cast<std::size_t>(worker) *
                        static_cast<std::size_t>(cols_) +
                    static_cast<std::size_t>(col);
                worker_next[offset] = next;
                next += local_counts[offset];
            }
        }
        std::vector<int> transpose_col_idx(values_.size());
        std::vector<double> transpose_values(values_.size());
#pragma omp parallel for schedule(static) num_threads(workers)
        for (int worker = 0; worker < workers; ++worker) {
            const int first = rows_ * worker / workers;
            const int last = rows_ * (worker + 1) / workers;
            int* next =
                worker_next.data() +
                static_cast<std::ptrdiff_t>(worker) * cols_;
            for (int row = first; row < last; ++row) {
                const int begin = row_ptr_[static_cast<std::size_t>(row)];
                const int end = row_ptr_[static_cast<std::size_t>(row) + 1U];
                for (int position = begin; position < end; ++position) {
                    const int transpose_row =
                        col_idx_[static_cast<std::size_t>(position)];
                    const int target = next[transpose_row]++;
                    transpose_col_idx[static_cast<std::size_t>(target)] = row;
                    transpose_values[static_cast<std::size_t>(target)] =
                        values_[static_cast<std::size_t>(position)];
                }
            }
        }
        return SparseMatrix(
            cols_, rows_, std::move(transpose_row_ptr),
            std::move(transpose_col_idx), std::move(transpose_values));
    }
    std::vector<int> transpose_row_ptr(static_cast<std::size_t>(cols_) + 1U, 0);
    for (int col : col_idx_) {
        ++transpose_row_ptr[static_cast<std::size_t>(col) + 1U];
    }
    for (int row = 0; row < cols_; ++row) {
        transpose_row_ptr[static_cast<std::size_t>(row) + 1U] +=
            transpose_row_ptr[static_cast<std::size_t>(row)];
    }
    std::vector<int> next = transpose_row_ptr;
    std::vector<int> transpose_col_idx(values_.size());
    std::vector<double> transpose_values(values_.size());
    for (int row = 0; row < rows_; ++row) {
        for (int position = row_ptr_[static_cast<std::size_t>(row)];
             position < row_ptr_[static_cast<std::size_t>(row) + 1U]; ++position) {
            const int transpose_row = col_idx_[static_cast<std::size_t>(position)];
            const int target = next[static_cast<std::size_t>(transpose_row)]++;
            transpose_col_idx[static_cast<std::size_t>(target)] = row;
            transpose_values[static_cast<std::size_t>(target)] =
                values_[static_cast<std::size_t>(position)];
        }
    }
    return SparseMatrix(cols_, rows_, std::move(transpose_row_ptr),
                        std::move(transpose_col_idx),
                        std::move(transpose_values));
}
inline Vector SparseMatrix::diagonal() const {
    const int n = std::min(rows_, cols_);
    Vector result(static_cast<std::size_t>(n), 0.0);
    for (int row = 0; row < n; ++row) {
        for (int pos = row_ptr_[static_cast<std::size_t>(row)];
             pos < row_ptr_[static_cast<std::size_t>(row) + 1U]; ++pos) {
            if (col_idx_[static_cast<std::size_t>(pos)] == row) {
                result[static_cast<std::size_t>(row)] = values_[static_cast<std::size_t>(pos)];
                break;
            }
        }
    }
    return result;
}
inline void SparseCholesky::factorize(
    const SparseMatrix& matrix,
    const std::vector<int>& new_to_old_permutation) {
    n_ = matrix.rows();
    new_to_old_ = new_to_old_permutation;
    std::vector<int> old_to_new(static_cast<std::size_t>(n_), -1);
    for (int index = 0; index < n_; ++index) {
        const int old = new_to_old_[static_cast<std::size_t>(index)];
        old_to_new[static_cast<std::size_t>(old)] = index;
    }
    std::vector<int> c_column_ptr(static_cast<std::size_t>(n_) + 1U, 0);
    for (int old_row = 0; old_row < n_; ++old_row) {
        const int new_row = old_to_new[static_cast<std::size_t>(old_row)];
        for (int position = matrix.row_ptr()[static_cast<std::size_t>(old_row)];
             position < matrix.row_ptr()[static_cast<std::size_t>(old_row) + 1U];
             ++position) {
            const int old_col = matrix.col_idx()[static_cast<std::size_t>(position)];
            const int new_col = old_to_new[static_cast<std::size_t>(old_col)];
            if (new_row <= new_col) {
                ++c_column_ptr[static_cast<std::size_t>(new_col) + 1U];
            }
        }
    }
    for (int col = 0; col < n_; ++col) {
        c_column_ptr[static_cast<std::size_t>(col) + 1U] +=
            c_column_ptr[static_cast<std::size_t>(col)];
    }
    std::vector<int> c_next = c_column_ptr;
    std::vector<int> c_row_idx(
        static_cast<std::size_t>(c_column_ptr.back()));
    std::vector<double> c_values(
        static_cast<std::size_t>(c_column_ptr.back()));
    for (int old_row = 0; old_row < n_; ++old_row) {
        const int new_row = old_to_new[static_cast<std::size_t>(old_row)];
        for (int position = matrix.row_ptr()[static_cast<std::size_t>(old_row)];
             position < matrix.row_ptr()[static_cast<std::size_t>(old_row) + 1U];
             ++position) {
            const int old_col = matrix.col_idx()[static_cast<std::size_t>(position)];
            const int new_col = old_to_new[static_cast<std::size_t>(old_col)];
            if (new_row <= new_col) {
                const int target = c_next[static_cast<std::size_t>(new_col)]++;
                c_row_idx[static_cast<std::size_t>(target)] = new_row;
                c_values[static_cast<std::size_t>(target)] =
                    matrix.values()[static_cast<std::size_t>(position)];
            }
        }
    }
    std::vector<std::pair<int, double>> entries;
    for (int col = 0; col < n_; ++col) {
        const int begin = c_column_ptr[static_cast<std::size_t>(col)];
        const int end = c_column_ptr[static_cast<std::size_t>(col) + 1U];
        entries.clear();
        entries.reserve(static_cast<std::size_t>(end - begin));
        for (int position = begin; position < end; ++position) {
            entries.emplace_back(
                c_row_idx[static_cast<std::size_t>(position)],
                c_values[static_cast<std::size_t>(position)]);
        }
        std::sort(entries.begin(), entries.end());
        for (int offset = 0; offset < end - begin; ++offset) {
            c_row_idx[static_cast<std::size_t>(begin + offset)] =
                entries[static_cast<std::size_t>(offset)].first;
            c_values[static_cast<std::size_t>(begin + offset)] =
                entries[static_cast<std::size_t>(offset)].second;
        }
    }
    std::vector<int> parent(static_cast<std::size_t>(n_), -1);
    std::vector<int> ancestor(static_cast<std::size_t>(n_), -1);
    for (int col = 0; col < n_; ++col) {
        for (int position = c_column_ptr[static_cast<std::size_t>(col)];
             position < c_column_ptr[static_cast<std::size_t>(col) + 1U];
             ++position) {
            int row = c_row_idx[static_cast<std::size_t>(position)];
            if (row >= col) continue;
            while (row != -1 && row < col) {
                const int next = ancestor[static_cast<std::size_t>(row)];
                ancestor[static_cast<std::size_t>(row)] = col;
                if (next == -1) parent[static_cast<std::size_t>(row)] = col;
                row = next;
            }
        }
    }
    std::vector<std::vector<std::pair<int, double>>> lower_columns(
        static_cast<std::size_t>(n_));
    std::vector<double> work(static_cast<std::size_t>(n_), 0.0);
    std::vector<int> visited(static_cast<std::size_t>(n_), -1);
    std::vector<int> stack(static_cast<std::size_t>(n_));
    std::vector<int> path;
    path.reserve(static_cast<std::size_t>(n_));
    std::vector<int> lower_column_sizes(
        static_cast<std::size_t>(n_), 1);
    for (int col = 0; col < n_; ++col) {
        int top = n_;
        visited[static_cast<std::size_t>(col)] = col;
        for (int position = c_column_ptr[static_cast<std::size_t>(col)];
             position < c_column_ptr[static_cast<std::size_t>(col) + 1U];
             ++position) {
            int row = c_row_idx[static_cast<std::size_t>(position)];
            if (row >= col) continue;
            path.clear();
            while (row != -1 &&
                   visited[static_cast<std::size_t>(row)] != col) {
                path.push_back(row);
                visited[static_cast<std::size_t>(row)] = col;
                row = parent[static_cast<std::size_t>(row)];
            }
            while (!path.empty()) {
                stack[static_cast<std::size_t>(--top)] = path.back();
                path.pop_back();
            }
        }
        for (int position = top; position < n_; ++position) {
            ++lower_column_sizes[static_cast<std::size_t>(
                stack[static_cast<std::size_t>(position)])];
        }
    }
    for (int col = 0; col < n_; ++col) {
        lower_columns[static_cast<std::size_t>(col)].reserve(
            static_cast<std::size_t>(
                lower_column_sizes[static_cast<std::size_t>(col)]));
    }
    std::fill(visited.begin(), visited.end(), -1);
    for (int col = 0; col < n_; ++col) {
        int top = n_;
        visited[static_cast<std::size_t>(col)] = col;
        for (int position = c_column_ptr[static_cast<std::size_t>(col)];
             position < c_column_ptr[static_cast<std::size_t>(col) + 1U];
             ++position) {
            int row = c_row_idx[static_cast<std::size_t>(position)];
            if (row >= col) continue;
            path.clear();
            while (row != -1 && visited[static_cast<std::size_t>(row)] != col) {
                path.push_back(row);
                visited[static_cast<std::size_t>(row)] = col;
                row = parent[static_cast<std::size_t>(row)];
            }
            while (!path.empty()) {
                stack[static_cast<std::size_t>(--top)] = path.back();
                path.pop_back();
            }
        }
        for (int position = c_column_ptr[static_cast<std::size_t>(col)];
             position < c_column_ptr[static_cast<std::size_t>(col) + 1U];
             ++position) {
            work[static_cast<std::size_t>(
                c_row_idx[static_cast<std::size_t>(position)])] =
                c_values[static_cast<std::size_t>(position)];
        }
        double diagonal = work[static_cast<std::size_t>(col)];
        work[static_cast<std::size_t>(col)] = 0.0;
        for (int position = top; position < n_; ++position) {
            const int pivot = stack[static_cast<std::size_t>(position)];
            const auto& lower_column =
                lower_columns[static_cast<std::size_t>(pivot)];
            if (lower_column.empty()) {
                throw std::runtime_error("SparseCholesky symbolic factorization failed");
            }
            const double value =
                work[static_cast<std::size_t>(pivot)] /
                lower_column.front().second;
            work[static_cast<std::size_t>(pivot)] = 0.0;
            for (std::size_t entry = 1; entry < lower_column.size(); ++entry) {
                work[static_cast<std::size_t>(lower_column[entry].first)] -=
                    lower_column[entry].second * value;
            }
            diagonal -= value * value;
            lower_columns[static_cast<std::size_t>(pivot)].emplace_back(
                col, value);
        }
        if (!(diagonal > 0.0) || !std::isfinite(diagonal)) {
            throw std::runtime_error("SparseCholesky: matrix is not SPD");
        }
        lower_columns[static_cast<std::size_t>(col)].emplace_back(
            col, std::sqrt(diagonal));
    }
    column_ptr_.assign(static_cast<std::size_t>(n_) + 1U, 0);
    for (int col = 0; col < n_; ++col) {
        column_ptr_[static_cast<std::size_t>(col) + 1U] =
            column_ptr_[static_cast<std::size_t>(col)] +
            static_cast<int>(lower_columns[static_cast<std::size_t>(col)].size());
    }
    row_idx_.resize(static_cast<std::size_t>(column_ptr_.back()));
    values_.resize(static_cast<std::size_t>(column_ptr_.back()));
    for (int col = 0; col < n_; ++col) {
        int target = column_ptr_[static_cast<std::size_t>(col)];
        for (const auto& entry : lower_columns[static_cast<std::size_t>(col)]) {
            row_idx_[static_cast<std::size_t>(target)] = entry.first;
            values_[static_cast<std::size_t>(target)] = entry.second;
            ++target;
        }
    }
}
inline void SparseCholesky::solve(const Vector& rhs, Vector& result,
                                  Vector& work) const {
    work.resize(static_cast<std::size_t>(n_));
    for (int index = 0; index < n_; ++index) {
        work[static_cast<std::size_t>(index)] =
            rhs[static_cast<std::size_t>(
                new_to_old_[static_cast<std::size_t>(index)])];
    }
    for (int col = 0; col < n_; ++col) {
        const int begin = column_ptr_[static_cast<std::size_t>(col)];
        const double diagonal = values_[static_cast<std::size_t>(begin)];
        work[static_cast<std::size_t>(col)] /= diagonal;
        for (int position = begin + 1;
             position < column_ptr_[static_cast<std::size_t>(col) + 1U];
             ++position) {
            work[static_cast<std::size_t>(
                row_idx_[static_cast<std::size_t>(position)])] -=
                values_[static_cast<std::size_t>(position)] *
                work[static_cast<std::size_t>(col)];
        }
    }
    for (int col = n_ - 1; col >= 0; --col) {
        const int begin = column_ptr_[static_cast<std::size_t>(col)];
        for (int position = begin + 1;
             position < column_ptr_[static_cast<std::size_t>(col) + 1U];
             ++position) {
            work[static_cast<std::size_t>(col)] -=
                values_[static_cast<std::size_t>(position)] *
                work[static_cast<std::size_t>(
                    row_idx_[static_cast<std::size_t>(position)])];
        }
        work[static_cast<std::size_t>(col)] /=
            values_[static_cast<std::size_t>(begin)];
    }
    result.resize(static_cast<std::size_t>(n_));
    for (int index = 0; index < n_; ++index) {
        result[static_cast<std::size_t>(
            new_to_old_[static_cast<std::size_t>(index)])] =
            work[static_cast<std::size_t>(index)];
    }
}
}
