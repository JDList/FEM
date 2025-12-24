

#pragma once

#include <vector>
#include <string>
#include <sstream>
#include <cstddef>   
#include <stdexcept> 
#include <cmath>     

namespace linalg {


using Vec = std::vector<double>;
using Mat = std::vector<Vec>;
using size_type = std::size_t;

inline Vec zero_vec(size_type n) {
    return Vec(n, 0.0);
}


inline Mat zero_mat(size_type r, size_type c) {
    return Mat(r, Vec(c, 0.0));
}

inline std::string fmt2(const char* a, size_type x, const char* b, size_type y) {
    std::ostringstream os;
    os << a << x << b << y;
    return os.str();
}


inline void ensure_vec_size(const Vec &v, size_type n) {
    if (v.size() != n) {
        throw std::invalid_argument(fmt2("vector size mismatch: got ", v.size(), ", expected ", n));
    }
}


inline void ensure_mat_square(const Mat &m) {
    if (m.empty()) {
        throw std::invalid_argument("matrix is empty");
    }
    size_type n = m.size();
    for (size_type i = 0; i < n; ++i) {
        if (m[i].size() != n) {
            throw std::invalid_argument(fmt2("matrix not square: row ", i, " has length ", m[i].size()));
        }
    }
}


inline void ensure_mat_mul_sizes(const Mat &A, const Mat &B) {
    if (A.empty() || B.empty()) {
        throw std::invalid_argument("empty matrix supplied for multiplication");
    }
    size_type a_cols = A[0].size();
    size_type b_rows = B.size();
    if (a_cols != b_rows) {
        throw std::invalid_argument(fmt2("incompatible matmul dims: A cols=", a_cols, ", B rows=", b_rows));
    }
}

Vec add(const Vec &a, const Vec &b);


double dot(const Vec &a, const Vec &b);

Vec scalarmul(const Vec &v, double alpha);

Vec cross3(const Vec &a, const Vec &b);

Mat scalarmulmat(const Mat &m, double alpha);

Mat transpose(const Mat &A);

Mat matmul(const Mat &A, const Mat &B);

double determinant(Mat A);

void print_vec(const Vec &v);

void print_mat(const Mat &m);
} 

namespace fem {
    
using size_type = linalg::size_type;
using Vec = linalg::Vec;
using Mat = linalg::Mat;

double triangle_area(const Mat &coords);

Vec barycentric_coords(const Mat &coords, double x, double y);

Vec eval_shape_linear_tri(const Mat &coords, double x, double y); 

Mat grad_shape_linear_tri(const Mat &coords); 

Mat element_stiffness_tri(const Mat &coords, double kappa = 1.0);

Mat element_mass_tri_consistent(const Mat &coords, double density = 1.0);

Vec element_mass_tri_lumped(const Mat &coords, double density = 1.0); 

double interp_value_tri(const Mat &coords, const Vec &nodal_vals, double x, double y);

Mat element_stiffness_elastic(const Mat &coords,
                                      double E, double nu,
                                      bool plane_stress,
                                      double t = 1.0);

}
