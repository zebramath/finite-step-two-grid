#pragma once
#include "pde/diffusion_problem.hpp"
#include <array>
namespace tgi {
namespace energy_interpolation_detail {
struct GlobalFSystem {
    struct RowStencil {
        std::array<int, 5> columns{};
        std::array<double, 5> values{};
    };
    SparseMatrix matrix;
    std::vector<int> f_nodes;
    std::vector<int> local_index;
    Vector inverse_diagonal;
    std::vector<std::vector<std::pair<int, double>>> rhs_entries;
    std::vector<RowStencil> stencil;
    double row_product(std::size_t row, const Vector& x) const {
        double sum = 0.0;
        if (!stencil.empty()) {
            const auto& packed = stencil[row];
            for (std::size_t entry = 0; entry < 5U; ++entry) sum += packed.values[entry] * x[static_cast<std::size_t>(packed.columns[entry])];
        } else {
            for (int pos = matrix.row_ptr()[row]; pos < matrix.row_ptr()[row + 1U]; ++pos) sum += matrix.values()[static_cast<std::size_t>(pos)] * x[static_cast<std::size_t>(matrix.col_idx()[static_cast<std::size_t>(pos)])];
        }
        return sum;
    }
};
inline int coarse_id_from_fine_node(
    const StructuredGrid& grid, int fine) {
    const auto [ix, iy] = grid.fine_coords(fine);
    return grid.coarse_id(
        (ix + 1) / grid.ratio() - 1,
        (iy + 1) / grid.ratio() - 1);
}
inline GlobalFSystem assemble_global_f_system(
    const StructuredGrid& grid, const SparseMatrix& matrix) {
    GlobalFSystem system;
    system.f_nodes = grid.all_f_nodes();
    system.rhs_entries.resize(
        static_cast<std::size_t>(grid.coarse_size()));
    system.local_index.assign(
        static_cast<std::size_t>(grid.fine_size()), -1);
    for (std::size_t local = 0; local < system.f_nodes.size(); ++local) {
        system.local_index[static_cast<std::size_t>(system.f_nodes[local])] =
            static_cast<int>(local);
    }
    std::vector<int> row_ptr(system.f_nodes.size() + 1U, 0);
    std::vector<int> col_idx;
    Vector values;
    col_idx.reserve(system.f_nodes.size() * 5U);
    values.reserve(system.f_nodes.size() * 5U);
    for (std::size_t local_row = 0;
         local_row < system.f_nodes.size(); ++local_row) {
        const int global_row = system.f_nodes[local_row];
        for (int position =
                 matrix.row_ptr()[static_cast<std::size_t>(global_row)];
             position < matrix.row_ptr()[
                 static_cast<std::size_t>(global_row) + 1U];
             ++position) {
            const int global_col =
                matrix.col_idx()[static_cast<std::size_t>(position)];
            const double value =
                matrix.values()[static_cast<std::size_t>(position)];
            const int local_col =
                system.local_index[static_cast<std::size_t>(global_col)];
            if (local_col >= 0) {
                col_idx.push_back(local_col);
                values.push_back(value);
            } else if (grid.is_coarse_node(global_col)) {
                const int coarse =
                    coarse_id_from_fine_node(grid, global_col);
                system.rhs_entries[static_cast<std::size_t>(coarse)]
                    .push_back({static_cast<int>(local_row), -value});
            }
        }
        row_ptr[local_row + 1U] = static_cast<int>(values.size());
    }
    system.matrix = SparseMatrix(
        static_cast<int>(system.f_nodes.size()),
        static_cast<int>(system.f_nodes.size()), std::move(row_ptr),
        std::move(col_idx), std::move(values));
    bool short_rows = true;
    for (std::size_t row = 0; row < system.f_nodes.size(); ++row) short_rows = short_rows && system.matrix.row_ptr()[row + 1U] - system.matrix.row_ptr()[row] <= 5;
    if (short_rows) {
        system.stencil.resize(system.f_nodes.size());
        for (std::size_t row = 0; row < system.f_nodes.size(); ++row) {
            std::size_t entry = 0;
            for (int pos = system.matrix.row_ptr()[row]; pos < system.matrix.row_ptr()[row + 1U]; ++pos) {
                system.stencil[row].columns[entry] = system.matrix.col_idx()[static_cast<std::size_t>(pos)];
                system.stencil[row].values[entry++] = system.matrix.values()[static_cast<std::size_t>(pos)];
            }
        }
    }
    const Vector diagonal = system.matrix.diagonal();
    system.inverse_diagonal.resize(diagonal.size());
    for (std::size_t index = 0; index < diagonal.size(); ++index) {
        system.inverse_diagonal[index] = 1.0 / diagonal[index];
    }
    return system;
}
}
inline SparseMatrix build_geometric_interpolation(
    const StructuredGrid& grid) {
    std::vector<int> row_ptr(
        static_cast<std::size_t>(grid.fine_size()) + 1U, 0);
    std::vector<int> col_idx;
    Vector values;
    col_idx.reserve(static_cast<std::size_t>(4 * grid.fine_size()));
    values.reserve(static_cast<std::size_t>(4 * grid.fine_size()));
    const int coarse_intervals = grid.intervals() / grid.ratio();
    for (int fine = 0; fine < grid.fine_size(); ++fine) {
        const auto [ix, iy] = grid.fine_coords(fine);
        const int lattice_x = ix + 1;
        const int lattice_y = iy + 1;
        const int left_x = lattice_x / grid.ratio();
        const int left_y = lattice_y / grid.ratio();
        const double tx = static_cast<double>(lattice_x % grid.ratio()) /
            static_cast<double>(grid.ratio());
        const double ty = static_cast<double>(lattice_y % grid.ratio()) /
            static_cast<double>(grid.ratio());
        const int qx[2] = {left_x, left_x + 1};
        const int qy[2] = {left_y, left_y + 1};
        const double wx[2] = {1.0 - tx, tx};
        const double wy[2] = {1.0 - ty, ty};
        for (int ay = 0; ay < 2; ++ay) {
            for (int ax = 0; ax < 2; ++ax) {
                if (wx[ax] == 0.0 || wy[ay] == 0.0 ||
                    qx[ax] <= 0 || qx[ax] >= coarse_intervals ||
                    qy[ay] <= 0 || qy[ay] >= coarse_intervals) {
                    continue;
                }
                col_idx.push_back(
                    grid.coarse_id(qx[ax] - 1, qy[ay] - 1));
                values.push_back(wx[ax] * wy[ay]);
            }
        }
        row_ptr[static_cast<std::size_t>(fine) + 1U] =
            static_cast<int>(values.size());
    }
    return SparseMatrix(
        grid.fine_size(), grid.coarse_size(), std::move(row_ptr),
        std::move(col_idx), std::move(values));
}
}
