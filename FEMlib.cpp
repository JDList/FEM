#include "FEMlib.hpp"

#include <algorithm>   
#include <cmath>       
#include <stdexcept>   
#include <iostream>

namespace linalg {

using size_type = linalg::size_type;


Vec add(const Vec &a, const Vec &b) {
    if (a.size() != b.size()) throw std::invalid_argument("add: vector size mismatch");
    Vec out(a.size());
    for (size_type i = 0; i < a.size(); ++i) out[i] = a[i] + b[i];
    return out;
}


double dot(const Vec &a, const Vec &b) {
    if (a.size() != b.size()) throw std::invalid_argument("dot: vector size mismatch");
    double s = 0.0;
    for (size_type i = 0; i < a.size(); ++i) s += a[i] * b[i];
    return s;
}


Vec scalarmul(const Vec &v, double alpha) {
    Vec out(v.size());
    for (size_type i = 0; i < v.size(); ++i) out[i] = v[i] * alpha;
    return out;
}


Vec cross3(const Vec &a, const Vec &b) {
    if (a.size() != 3 || b.size() != 3) throw std::invalid_argument("cross3 requires length-3 vectors");
    return Vec{ a[1]*b[2] - a[2]*b[1],
                a[2]*b[0] - a[0]*b[2],
                a[0]*b[1] - a[1]*b[0] };
}
Mat scalarmulmat(const Mat &m, double alpha){
    if (m.empty()) return Mat{};
    
    size_type r = m.size();
    size_type c = m[0].size();
    Mat out = zero_mat(c,r);
    for (size_type i = 0; i <r; ++i){
        for (size_type j = 0; j < c; ++j){
            out[i][j] = m[i][j]*alpha;
        }
    }
    return m;
}

Mat transpose(const Mat &A) {
    if (A.empty()) return Mat{};
    size_type r = A.size();
    size_type c = A[0].size();
    
    for (size_type i = 0; i < r; ++i) {
        if (A[i].size() != c) throw std::invalid_argument("transpose: ragged matrix");
    }
    Mat T = zero_mat(c, r);
    for (size_type i = 0; i < r; ++i) {
        for (size_type j = 0; j < c; ++j) {
            T[j][i] = A[i][j];
        }
    }
    return T;
}


Mat matmul(const Mat &A, const Mat &B) {
    if (A.empty() || B.empty()) throw std::invalid_argument("matmul: empty matrix");
    size_type r = A.size();
    size_type m = A[0].size();
    size_type b_rows = B.size();
    if (m != b_rows) throw std::invalid_argument("matmul: incompatible dims (A.cols != B.rows)");
    size_type c = B[0].size();
    
    for (size_type i = 0; i < B.size(); ++i) {
        if (B[i].size() != c) throw std::invalid_argument("matmul: ragged matrix B");
    }
    Mat C = zero_mat(r, c);
    
    for (size_type i = 0; i < r; ++i) {
        if (A[i].size() != m) throw std::invalid_argument("matmul: ragged matrix A");
        for (size_type k = 0; k < m; ++k) {
            double aik = A[i][k];
            if (aik == 0.0) continue;
            for (size_type j = 0; j < c; ++j) {
                C[i][j] += aik * B[k][j];
            }
        }
    }
    return C;
}



double determinant(Mat A) {
    ensure_mat_square(A);            
    size_type n = A.size();
    if (n == 0) return 1.0;          

    const double EPS = 1e-15;
    double det_sign = 1.0;
    double det_abs = 1.0;

    for (size_type k = 0; k < n; ++k) {
        
        size_type piv = k;
        double maxv = std::abs(A[k][k]);
        for (size_type i = k + 1; i < n; ++i) {
            double v = std::abs(A[i][k]);
            if (v > maxv) { maxv = v; piv = i; }
        }
        if (maxv < EPS) {
            
            return 0.0;
        }
        if (piv != k) {
            std::swap(A[k], A[piv]);
            det_sign = -det_sign;
        }
        double pivot = A[k][k];
        det_abs *= pivot;

        
        for (size_type i = k + 1; i < n; ++i) {
            double factor = A[i][k];
            A[i][k] = 0.0;
            for (size_type j = k + 1; j < n; ++j) {
                A[i][j] -= factor * A[k][j];
            }
        }
    }
    return det_sign * det_abs;
}
void print_vec(const Vec &v){
    std::cout << "[";
    for(size_t i=0;i<v.size();++i){
        if(i) std::cout << ", ";
        std::cout << v[i];
    }
     std::cout << "]\n";
}
void print_mat(const Mat &M){
     std::cout << "[\n";
    for(const auto &row: M){
        std::cout << "  ";
        print_vec(row);
        std::cout << "\n";
    }
    std::cout << "]\n";
}


} 


namespace fem {

using size_type = linalg::size_type;
using Vec = linalg::Vec;
using Mat = linalg::Mat;



double triangle_area(const Mat &coords) {
    if (coords.size() != 3 || coords[0].size() < 2) 
        throw std::invalid_argument("triangle_area: coords must be 3x2");
    double x0 = coords[0][0], y0 = coords[0][1];
    double x1 = coords[1][0], y1 = coords[1][1];
    double x2 = coords[2][0], y2 = coords[2][1];
    return 0.5 * ( (x1 - x0)*(y2 - y0) - (x2 - x0)*(y1 - y0) );
}



Vec barycentric_coords(const Mat &coords, double x, double y) {
    if (coords.size() != 3 || coords[0].size() < 2)
        throw std::invalid_argument("barycentric_coords: coords must be 3x2");
    double x0 = coords[0][0], y0 = coords[0][1];
    double x1 = coords[1][0], y1 = coords[1][1];
    double x2 = coords[2][0], y2 = coords[2][1];

    double detT = (x1 - x0)*(y2 - y0) - (x2 - x0)*(y1 - y0);
    if (std::abs(detT) < 1e-20) throw std::runtime_error("barycentric_coords: degenerate triangle");

    
    double l0 = ((x1 - x)*(y2 - y) - (x2 - x)*(y1 - y));
    double l1 = ((x2 - x)*(y0 - y) - (x0 - x)*(y2 - y));
    double l2 = 1.0 - l0 - l1;
    return Vec{l0, l1, l2};
}



Vec eval_shape_linear_tri(const Mat &coords, double x, double y) {
    return barycentric_coords(coords, x, y);
}



Mat grad_shape_linear_tri(const Mat &coords) {
    if (coords.size() != 3 || coords[0].size() < 2)
        throw std::invalid_argument("grad_shape_linear_tri: coords must be 3x2");

    double x0 = coords[0][0], y0 = coords[0][1];
    double x1 = coords[1][0], y1 = coords[1][1];
    double x2 = coords[2][0], y2 = coords[2][1];

    double detT = (x1 - x0)*(y2 - y0) - (x2 - x0)*(y1 - y0);
    if (std::abs(detT) < 1e-20) throw std::runtime_error("grad_shape_linear_tri: degenerate triangle");
    double inv_det = 1.0/detT;

    Mat grads = linalg::zero_mat(3, 2);

    grads[0][0] =  (y1 - y2) * 0.5 * inv_det * 2.0; 
    grads[0][1] =  (x2 - x1) * 0.5 * inv_det * 2.0; 
    grads[1][0] =  (y2 - y0) * 0.5 * inv_det * 2.0; 
    grads[1][1] =  (x0 - x2) * 0.5 * inv_det * 2.0; 
    grads[2][0] =  (y0 - y1) * 0.5 * inv_det * 2.0; 
    grads[2][1] =  (x1 - x0) * 0.5 * inv_det * 2.0; 
    
    for (size_type i = 0; i < 3; ++i) {
        grads[i][0] *= 1.0; 
        grads[i][1] *= 1.0;
    }
    
    
    grads[0][0] =  (y1 - y2) * inv_det;
    grads[0][1] =  (x2 - x1) * inv_det;
    grads[1][0] =  (y2 - y0) * inv_det;
    grads[1][1] =  (x0 - x2) * inv_det;
    grads[2][0] =  (y0 - y1) * inv_det;
    grads[2][1] =  (x1 - x0) * inv_det;

    return grads;
}



Mat element_stiffness_tri(const Mat &coords, double kappa ) {
    double area = std::abs(triangle_area(coords));
    if (area <= 0.0) throw std::runtime_error("element_stiffness_tri: non-positive area");

    Mat grads = grad_shape_linear_tri(coords); 
    Mat ke = linalg::zero_mat(3, 3);

    
    for (size_type i = 0; i < 3; ++i) {
        for (size_type j = 0; j < 3; ++j) {
            double dotg = grads[i][0]*grads[j][0] + grads[i][1]*grads[j][1];
            ke[i][j] = kappa * area * dotg;
        }
    }
    return ke;
}


Mat element_mass_tri_consistent(const Mat &coords, double density) {
    double area = std::abs(triangle_area(coords));
    if (area <= 0.0) throw std::runtime_error("element_mass_tri_consistent: non-positive area");

    
    double factor = density * area;
    Mat Me = linalg::zero_mat(3,3);
    for (size_type i = 0; i < 3; ++i) {
        for (size_type j = 0; j < 3; ++j) {
            Me[i][j] = factor * (i == j ? 2.0 : 1.0);
        }
    }
    return Me;
}


Vec element_mass_tri_lumped(const Mat &coords, double density) {
    double area = std::abs(triangle_area(coords));
    if (area <= 0.0) throw std::runtime_error("element_mass_tri_lumped: non-positive area");

    
    double m = density * area;
    return Vec{m, m, m};
}

double interp_value_tri(const Mat &coords, const Vec &nodal_vals, double x, double y) {
    if (nodal_vals.size() != 3) throw std::invalid_argument("interp_value_tri: nodal_vals must be length 3");
    Vec N = eval_shape_linear_tri(coords, x, y);
    return N[0]*nodal_vals[0] + N[1]*nodal_vals[1] + N[2]*nodal_vals[2];
}


Mat element_stiffness_elastic(const Mat &coords,
                                      double E, double nu,
                                      bool plane_stress,
                                      double t ) {
    if (coords.size() != 3 || coords[0].size() < 2)
        throw std::invalid_argument("element_stiffness_elastic_no_body: coords must be 3x2");

    
    double A = std::abs(fem::triangle_area(coords));
    if (A <= 0.0) throw std::runtime_error("element_stiffness_elastic_no_body: degenerate triangle");

    
    Mat D = linalg::zero_mat(3, 3);
    if (plane_stress) {
        double fac = E; 
        D[0][0] = fac * 1.0;                 D[0][1] = fac * nu;                 D[0][2] = 0.0;
        D[1][0] = fac * nu;                 D[1][1] = fac * 1.0;               D[1][2] = 0.0;
        D[2][0] = 0.0;                      D[2][1] = 0.0;                      D[2][2] = fac * (1.0 - nu); 
    } else { 
        double fac = E;
        D[0][0] = fac * (1.0 - nu);         D[0][1] = fac * nu;                D[0][2] = 0.0;
        D[1][0] = fac * nu;                 D[1][1] = fac * (1.0 - nu);        D[1][2] = 0.0;
        D[2][0] = 0.0;                      D[2][1] = 0.0;                      D[2][2] = fac * (1.0 - 2.0 * nu);
    }

    
    Mat grads = fem::grad_shape_linear_tri(coords);

    
    Mat B = linalg::zero_mat(3, 6);
    for (size_t i = 0; i < 3; ++i) {
        double dNdx = grads[i][0];
        double dNdy = grads[i][1];
        B[0][2*i + 0] = dNdx;  B[0][2*i + 1] = 0.0;
        B[1][2*i + 0] = 0.0;   B[1][2*i + 1] = dNdy;
        B[2][2*i + 0] = dNdy;  B[2][2*i + 1] = dNdx;
    }

    
    Mat tmp = linalg::matmul(D, B);      
    Mat Bt = linalg::transpose(B);       
    Mat BtDB = linalg::matmul(Bt, tmp);  

    
    for (size_t i = 0; i < BtDB.size(); ++i)
        for (size_t j = 0; j < BtDB[0].size(); ++j)
            BtDB[i][j] *= (A * t);

    return BtDB;
}





} 
