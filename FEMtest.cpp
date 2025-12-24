

#include "FEMlib.hpp"           
#include <eigen3/Eigen/Sparse>
#include <eigen3/Eigen/Dense>
#include <vector>
#include <array>
#include <iostream>
#include <cmath>
#include <stdexcept>



void applyDirichletStrong(Eigen::SparseMatrix<double> &K,
                          Eigen::VectorXd &F,
                          const std::vector<char> &is_dirichlet,
                          const Eigen::VectorXd &prescribed)
{
    const int Ndof = (int)K.rows();
    if ((int)K.cols() != Ndof) throw std::runtime_error("applyDirichletStrong: K must be square");
    if ((int)F.size() != Ndof) throw std::runtime_error("applyDirichletStrong: F size mismatch");
    if ((int)prescribed.size() != Ndof) throw std::runtime_error("applyDirichletStrong: prescribed size mismatch");

    std::vector<int> fixed;
    std::vector<int> is_fixed_index(Ndof, -1);
    for (int d = 0; d < Ndof; ++d) {
        if (is_dirichlet[d]) {
            is_fixed_index[d] = (int)fixed.size();
            fixed.push_back(d);
        }
    }
    if (fixed.empty()) return;

    K.makeCompressed();

    
    for (int col = 0; col < K.outerSize(); ++col) {
        if (!is_dirichlet[col]) continue; 
        double uj = prescribed[col];
        if (uj == 0.0) continue; 
        for (Eigen::SparseMatrix<double>::InnerIterator it(K, col); it; ++it) {
            int i = it.row();
            
            if (!is_dirichlet[i]) {
                F[i] -= it.value() * uj;
            }
        }
    }

    
    
    for (int col = 0; col < K.outerSize(); ++col) {
        for (Eigen::SparseMatrix<double>::InnerIterator it(K, col); it; ++it) {
            int row = it.row();
            if (is_dirichlet[row] || is_dirichlet[col]) {
                it.valueRef() = 0.0;
            }
        }
    }

    
    for (int d : fixed) {
        K.coeffRef(d, d) = 1.0;
    }

    for (int d : fixed) {
        F[d] = prescribed[d];
    }

    K.prune(0.0);
}


#include <fstream>
#include <string>
#include <iomanip> 

void postprocess_and_write_vtk(const std::string &filename,
                               const std::vector<std::array<double,2>> &nodes,
                               const std::vector<std::array<int,3>> &elements,
                               const Eigen::VectorXd &U,
                               double E, double nu, bool plane_stress, double t) {
    using size_type = fem::size_type;
    size_type Nnodes = nodes.size();
    size_type Nelems = elements.size();

    
    fem::Mat D = linalg::zero_mat(3,3);
    if (plane_stress) {
        double fac = E/(1- nu*nu);
        D[0][0] = fac * 1.0; D[0][1] = fac * nu; D[0][2] = 0.0;
        D[1][0] = fac * nu; D[1][1] = fac * 1.0; D[1][2] = 0.0;
        D[2][0] = 0.0; D[2][1] = 0.0; D[2][2] = fac * (1.0 - nu)/2.0;
    } else {
        double fac = E;
        D[0][0] = fac * (1.0 - nu); D[0][1] = fac * nu; D[0][2] = 0.0;
        D[1][0] = fac * nu; D[1][1] = fac * (1.0 - nu); D[1][2] = 0.0;
        D[2][0] = 0.0; D[2][1] = 0.0; D[2][2] = fac * (1.0 - 2.0 * nu);
    }

    
    std::vector<std::array<double,3>> disp_points(Nnodes);
    for (size_type n = 0; n < Nnodes; ++n) {
        double ux = U[(int)(2*n)];
        double uy = U[(int)(2*n + 1)];
        disp_points[n] = { nodes[n][0] + ux, nodes[n][1] + uy, 0.0 };
    }

    
    std::vector<std::array<double,3>> elem_strain(Nelems);
    std::vector<std::array<double,3>> elem_stress(Nelems);

    for (size_type e = 0; e < Nelems; ++e) {
        
        fem::Mat coords;
        coords.push_back(linalg::Vec{ nodes[elements[e][0]][0], nodes[elements[e][0]][1] });
        coords.push_back(linalg::Vec{ nodes[elements[e][1]][0], nodes[elements[e][1]][1] });
        coords.push_back(linalg::Vec{ nodes[elements[e][2]][0], nodes[elements[e][2]][1] });

        
        fem::Mat grads = fem::grad_shape_linear_tri(coords);

        
        fem::Mat B = linalg::zero_mat(3, 6);
        for (size_type i = 0; i < 3; ++i) {
            double dNdx = grads[i][0];
            double dNdy = grads[i][1];
            B[0][2*i + 0] = dNdx;  B[0][2*i + 1] = 0.0;
            B[1][2*i + 0] = 0.0;   B[1][2*i + 1] = dNdy;
            B[2][2*i + 0] = dNdy;  B[2][2*i + 1] = dNdx;
        }

        
        std::array<double,6> ue{};
        for (size_type i = 0; i < 3; ++i) {
            int gi = elements[e][i];
            ue[2*i + 0] = U[2*gi + 0];
            ue[2*i + 1] = U[2*gi + 1];
        }

        
        std::array<double,3> eps = {0.0, 0.0, 0.0};
        for (size_type r = 0; r < 3; ++r) {
            double s = 0.0;
            for (size_type c = 0; c < 6; ++c) s += B[r][c] * ue[c];
            eps[r] = s;
        }

        
        std::array<double,3> sigma = {0.0,0.0,0.0};
        for (size_type r = 0; r < 3; ++r) {
            double s = 0.0;
            for (size_type c = 0; c < 3; ++c) s += D[r][c] * eps[c];
            sigma[r] = s;
        }

        elem_strain[e] = eps;
        elem_stress[e] = sigma;
    }

    
    std::vector<double> nodal_sxx(Nnodes, 0.0);
    std::vector<int> nodal_count(Nnodes, 0);
    for (size_type e = 0; e < Nelems; ++e) {
        double sxx = elem_stress[e][0];
        for (int k = 0; k < 3; ++k) {
            int vi = elements[e][k];
            nodal_sxx[vi] += sxx;
            nodal_count[vi] += 1;
        }
    }
    for (size_type n = 0; n < Nnodes; ++n) {
        if (nodal_count[n] > 0) nodal_sxx[n] /= double(nodal_count[n]);
    }

    
    std::ofstream out(filename);
    if (!out) {
        std::cerr << "Failed to open " << filename << " for writing\n";
        return;
    }
    out << "# vtk DataFile Version 2.0\n";
    out << "FEM results (displaced points, displacements, cell stress)\n";
    out << "ASCII\n";
    out << "DATASET UNSTRUCTURED_GRID\n";

    
    out << "POINTS " << Nnodes << " float\n";
    out << std::setprecision(7);
    for (size_type n = 0; n < Nnodes; ++n) {
        out << disp_points[n][0] << " " << disp_points[n][1] << " " << disp_points[n][2] << "\n";
    }

    
    out << "CELLS " << Nelems << " " << (4 * Nelems) << "\n";
    for (size_type e = 0; e < Nelems; ++e) {
        out << 3 << " " << elements[e][0] << " " << elements[e][1] << " " << elements[e][2] << "\n";
    }

    
    out << "CELL_TYPES " << Nelems << "\n";
    for (size_type e = 0; e < Nelems; ++e) out << 5 << "\n";

    
    out << "POINT_DATA " << Nnodes << "\n";
    out << "VECTORS displacement float\n";
    for (size_type n = 0; n < Nnodes; ++n) {
        double ux = U[(int)(2*n)];
        double uy = U[(int)(2*n + 1)];
        out << ux << " " << uy << " " << 0.0 << "\n";
    }
    out << "SCALARS nodal_sxx float 1\n";
    out << "LOOKUP_TABLE default\n";
    for (size_type n = 0; n < Nnodes; ++n) out << nodal_sxx[n] << "\n";

    
    out << "CELL_DATA " << Nelems << "\n";
    out << "TENSORS stress float\n";
    for (size_type e = 0; e < Nelems; ++e) {
        double sxx = elem_stress[e][0];
        double syy = elem_stress[e][1];
        double sxy = elem_stress[e][2];
        
        out << sxx << " " << sxy << " 0\n";
        out << sxy << " " << syy << " 0\n";
        out << 0.0 << " " << 0.0 << " 0\n";
    }

    out.close();
    std::cout << "Wrote VTK file: " << filename << "\n";
}



#include <unordered_map>
#include <cstdint>


void split_each_triangle_in_two(
    std::vector<std::array<double,2>> &nodes_in,
    std::vector<std::array<int,3>> &elems_in)
{
    if (nodes_in.empty() && !elems_in.empty())
        throw std::invalid_argument("split_each_triangle_in_two: nodes empty but elements present");

    
    std::vector<std::array<double,2>> nodes_out = nodes_in;
    std::vector<std::array<int,3>> elems_out;
    elems_out.reserve(elems_in.size() * 2);

    
    std::unordered_map<uint64_t, int> edge_mid;
    edge_mid.reserve(elems_in.size() * 3 / 2);

    auto edge_key = [](int a, int b) -> uint64_t {
        uint32_t u = static_cast<uint32_t>(std::min(a,b));
        uint32_t v = static_cast<uint32_t>(std::max(a,b));
        return (static_cast<uint64_t>(u) << 32) | static_cast<uint64_t>(v);
    };

    
    for (const auto &tri : elems_in) {
        int a = tri[0];
        int b = tri[1];
        int c = tri[2];

        if (a < 0 || b < 0 || c < 0) throw std::invalid_argument("split_each_triangle_in_two: negative index");
        if (static_cast<size_t>(a) >= nodes_in.size() ||
            static_cast<size_t>(b) >= nodes_in.size() ||
            static_cast<size_t>(c) >= nodes_in.size())
            throw std::invalid_argument("split_each_triangle_in_two: index out of range");

        
        auto sqdist = [&](int i, int j) -> double {
            double dx = nodes_in[i][0] - nodes_in[j][0];
            double dy = nodes_in[i][1] - nodes_in[j][1];
            return dx*dx + dy*dy;
        };

        double d_ab = sqdist(a,b);
        double d_bc = sqdist(b,c);
        double d_ca = sqdist(c,a);

        
        int i, j, k;
        if (d_ab >= d_bc && d_ab >= d_ca) { i = a; j = b; k = c; }
        else if (d_bc >= d_ab && d_bc >= d_ca) { i = b; j = c; k = a; }
        else { i = c; j = a; k = b; }

        
        uint64_t key = edge_key(i, j);
        int m;
        auto it = edge_mid.find(key);
        if (it != edge_mid.end()) {
            m = it->second;
        } else {
            std::array<double,2> mid;
            mid[0] = 0.5 * (nodes_in[i][0] + nodes_in[j][0]);
            mid[1] = 0.5 * (nodes_in[i][1] + nodes_in[j][1]);
            m = static_cast<int>(nodes_out.size());
            nodes_out.push_back(mid);
            edge_mid.emplace(key, m);
        }

        
        std::array<int,3> t1 = { i, m, k };
        std::array<int,3> t2 = { m, j, k };
        elems_out.push_back(t1);
        elems_out.push_back(t2);
    }

    
    nodes_in = std::move(nodes_out);
    elems_in = std::move(elems_out);
}
int main() {
    using namespace std;
    using size_type = linalg::size_type;
    /* L-shape
    vector<array<double,2>> nodes = {
        {0.0,0.0}, {0.0,1.0}, {1.0,1.0}, {1.0,0.0},
        {2.0,1.0}, {2.0,0.0}, {3.0,1.0}, {3.0,0.0},
        {4.0,1.0}, {4.0,0.0}, {5.0,1.0}, {5.0,0.0},
        {4.0,2.0}, {5.0,2.0}, {4.0,3.0}, {5.0,3.0},
        {4.0,4.0}, {5.0,4.0}, {4.0,5.0}, {5.0,5.0},
    };

    vector<array<int,3>> elements = {
        {0,1,2}, {0,2,3}, {2,3,4}, {3,4,5},
        {5,4,6}, {5,7,6}, {7,6,8}, {7,8,9},
        {9,8,10}, {9,10,11}, {8,12,13}, {8,13,10},
        {12,14,15}, {12,13,15}, {14,16,17}, {14,15,17},
        {16,18,19}, {16,17,19} 
    };
/* Square
    vector<array<double,2>> nodes = {
        {0.0,0.0}, {0.0,5.0}, {5.0,0.0}, {5.0,5.0}
    };

    vector<array<int,3>> elements = {
        {0,1,3}, {0,2,3}
    };
*/

    vector<array<double,2>> nodes;
    vector<array<int,3>> elements;
    double radius = 5.0;
    double width = 2.0;
    size_type num_per_curve = 100;
    nodes.reserve(num_per_curve*2);
    elements.reserve(num_per_curve*2);
    const double PI = 3.14159265358;

    for (size_type i = 0; i < num_per_curve; ++i){
        nodes.emplace_back(std::array<double,2>{radius*(1.0-std::cos(PI*((double)(i)/(double)(num_per_curve-1)))), radius*std::sin(PI*((double)(i)/(double)(num_per_curve-1)))});
    }

    for (size_type j = 0; j < num_per_curve; ++j){
        nodes.emplace_back(std::array<double,2>{width+(radius-width)*(1.0-std::cos(PI*((double)(j)/(double)(num_per_curve-1)))), (radius-width)*std::sin(PI*((double)(j)/(double)(num_per_curve-1)))});
    }

    for (size_type k = 0; k < num_per_curve - 1; ++k){
        elements.emplace_back(std::array<int,3>{(int)(k),(int)(k+1),(int)(num_per_curve+k)});
        elements.emplace_back(std::array<int,3>{(int)(num_per_curve+k),(int)(num_per_curve+k+1), (int)(k+1)});
    }

    split_each_triangle_in_two(nodes, elements);
    split_each_triangle_in_two(nodes, elements);
    split_each_triangle_in_two(nodes, elements);
    split_each_triangle_in_two(nodes, elements);

    double E = 10000000.0;
    double nu = 0.25;
    bool plane_stress = true;
    double t = 1.0;

    const size_type Nnodes = nodes.size();
    const size_type Ndof = 2 * Nnodes; 

    
    
    
    using Triplet = Eigen::Triplet<double>;
    vector<Triplet> triplets;
    triplets.reserve(elements.size() * 36); 

    Eigen::VectorXd F = Eigen::VectorXd::Zero((Eigen::Index)Ndof); 

    for (size_type e = 0; e < elements.size(); ++e) {
        
        linalg::Mat coords;
        coords.push_back(linalg::Vec{ nodes[elements[e][0]][0], nodes[elements[e][0]][1] });
        coords.push_back(linalg::Vec{ nodes[elements[e][1]][0], nodes[elements[e][1]][1] });
        coords.push_back(linalg::Vec{ nodes[elements[e][2]][0], nodes[elements[e][2]][1] });

        
        linalg::Mat Ke = fem::element_stiffness_elastic(coords, E, nu, plane_stress, t);

        
        for (size_type i = 0; i < 3; ++i) {
            int gi_u = 2 * elements[e][i];
            int gi_v = gi_u + 1;
            for (size_type j = 0; j < 3; ++j) {
                int gj_u = 2 * elements[e][j];
                int gj_v = gj_u + 1;

                double k00 = Ke[2*i + 0][2*j + 0];
                double k01 = Ke[2*i + 0][2*j + 1];
                double k10 = Ke[2*i + 1][2*j + 0];
                double k11 = Ke[2*i + 1][2*j + 1];

                triplets.emplace_back(gi_u, gj_u, k00);
                triplets.emplace_back(gi_u, gj_v, k01);
                triplets.emplace_back(gi_v, gj_u, k10);
                triplets.emplace_back(gi_v, gj_v, k11);
            }
        }
    }

    Eigen::SparseMatrix<double> K((Eigen::Index)Ndof, (Eigen::Index)Ndof);
    K.setFromTriplets(triplets.begin(), triplets.end());

    
    
    
    std::vector<char> is_dirichlet(Ndof, 0);
    Eigen::VectorXd prescribed = Eigen::VectorXd::Zero((Eigen::Index)Ndof);
    const double tol = 1e-3;

    
    for (size_t n = 0; n < nodes.size(); ++n) {
        if (std::abs(nodes[n][0] - 0.0) < tol) {
            is_dirichlet[2*n]     = 1; prescribed[2*n]     = 0.0;
            is_dirichlet[2*n + 1] = 1; prescribed[2*n + 1] = 0.0;
        }
    }

    
    for (size_t n = 0; n < nodes.size(); ++n) {
        if ((std::abs(nodes[n][1] - 0.0) < tol)&&(std::abs(nodes[n][0] - 10.0) <tol)) {
            int dof_u = 2 * static_cast<int>(n);
            if (!is_dirichlet[dof_u]) {
                is_dirichlet[dof_u] = 1;
                prescribed[dof_u]   = 1.0;
                prescribed[dof_u+1] = 0.0;
            }
        }
    }
    
    applyDirichletStrong(K, F, is_dirichlet, prescribed);
    
    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver;
    solver.compute(K);
    if (solver.info() != Eigen::Success) {
        std::cerr << "Factorization failed\n";
        return -1;
    }
    Eigen::VectorXd U = solver.solve(F);
    if (solver.info() != Eigen::Success) {
        std::cerr << "Solve failed\n";
        return -1;
    }
    
    cout << "NodeID, x, y, u, v\n";
    for (size_type n = 0; n < Nnodes; ++n) {
        double ux = U[(int)(2*n)];
        double uy = U[(int)(2*n + 1)];
        cout << n << ", " << nodes[n][0] << ", " << nodes[n][1] << ", "
             << ux << ", " << uy << "\n";
    }


    postprocess_and_write_vtk("results.vtk", nodes, elements, U, E, nu, plane_stress, t);
    return 0;
}
