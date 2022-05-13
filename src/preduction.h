// References:
//
// [1] Borno, M. et.al. "On ‘decorrelation’ in solving integer least-squares 
//     problems for ambiguity determination", Survey Review, 46(334):37-49,
//     2014.

#ifndef BORNO_PREDUCTION_H__
#define BORNO_PREDUCTION_H__

#include "ltdl_min_pivot.h"

#include <eigen3/Eigen/Core>

#include <cmath>
#include <iostream>

// Implements [1] Algorithm 1 "Gauss".
//
// Integer Gauss Transformations):
// Given:
// - A unit lower triangle L \in Re{NxN}
// - Index pair (i,j)
// - Real LS estimate x_hat \in Re{N}
// - Integer permutation, Z \in Z{NxN}
//
// This algorithm first applies Z(i,j) to L such that |LZ(i,j)| <= 1/2,
// then computes Z(i,j)^T*x_hat and ZZ(i,j) which overwrite  x_hat and
// z respectively.
//
// Returns the value used for transformation.  If zero, no transformation
// was performed.
template <typename DerivedL, typename Index, typename DerivedX, 
          typename DerivedZ>
int IntegerGaussTransform(Eigen::MatrixBase<DerivedL> const &l_in,
                           const Index i, 
                           const Index j,
                           Eigen::MatrixBase<DerivedX> const &x_hat_in,
                           Eigen::MatrixBase<DerivedZ> const &z_in)
{
    std::cout << "Starting IGS" << std::endl;

    using Scalar = typename DerivedL::Scalar;

    Eigen::MatrixBase<DerivedL> &l = const_cast<Eigen::MatrixBase<DerivedL>&>(l_in);
    Eigen::MatrixBase<DerivedZ> &z = const_cast<Eigen::MatrixBase<DerivedZ>&>(z_in);
    Eigen::MatrixBase<DerivedX> &x_hat = const_cast<Eigen::MatrixBase<DerivedX>&>(x_hat_in);
    
    const int mu = std::lround(l.coeff(i,j));
    if (mu != 0)
    {
        const int tail_size = l.rows() - i;
        const Scalar scalar_mu = static_cast<Scalar>(mu);
        l.col(j).tail(tail_size).noalias() -= 
                scalar_mu * l.col(i).tail(tail_size);
        z.col(j).noalias() -= mu * z.col(i);
        x_hat.coeffRef(j) -= scalar_mu * x_hat.coeff(i);
    }

    return mu;
}

// Implements [1] Algorithm 2 "Permute".
//
// Given the L and D factors of the L^T*D*L factorisation of w_x in Re{NxN}, 
// index k, scalar delta , and Z in Z{NxN} which is equal to d(k+1) in Eq.(12),
// x. This algorithm computes the updated L and D factors in (11) after
// rows and columns `k` and `k+1` are interchanged. It also interchages
// entries of k and k+1 of x_hat and columns k and k+1 of Z.
template <typename DerivedL, typename DerivedD, typename Index,
          typename Scalar, typename DerivedX, typename DerivedZ>
void Permute(Eigen::MatrixBase<DerivedL> const &l_in,
             Eigen::MatrixBase<DerivedD> const &d_in, 
             const Index k,
             const Scalar delta, 
             Eigen::MatrixBase<DerivedX> const &x_hat_in, 
             Eigen::MatrixBase<DerivedZ> const &z_in)
{
    std::cout << "Starting Permute" << std::endl;
    Eigen::MatrixBase<DerivedL> &l = const_cast<Eigen::MatrixBase<DerivedL>&>(l_in);
    Eigen::MatrixBase<DerivedZ> &z = const_cast<Eigen::MatrixBase<DerivedZ>&>(z_in);
    Eigen::MatrixBase<DerivedX> &x_hat = const_cast<Eigen::MatrixBase<DerivedX>&>(x_hat_in);
    Eigen::MatrixBase<DerivedD> &d = const_cast<Eigen::MatrixBase<DerivedD>&>(d_in);
    
    const Index kp1 = k+1;

    // See [1] Eq.(12).
    const Scalar eta = d.coeff(k) / delta;

    // See [1] Eq.(13).
    const Scalar lambda = d.coeff(kp1) * l.coeff(kp1, k) / delta;
    
    // See [1] Eq.(12).
    d.coeffRef(k) = eta * d.coeff(kp1);
    d.coeffRef(kp1) = delta;

    // See [1] Eq.(14).
    Eigen::Matrix<Scalar, 2, 2> factor;
    factor << 
        -l.coeff(kp1, k),      1,
                     eta, lambda;
    l.block(k, 0, 2, k) *= factor;

    // See [1] Eq.(15).
    // Swap columns L(k+2:n, k) and L(k+2:n, k+1).
    const int l_swap_len = l.rows() - 2 - k;
    l.col(k).tail(l_swap_len).swap(l.col(kp1).tail(l_swap_len));

    // Swap columns Z(:,k) and Z(:,k+1).
    z.col(k).swap(z.col(kp1));

    // Swap entries x_hat(k) and x_hat(k+1).
    std::swap(x_hat.coeffRef(k), x_hat.coeffRef(kp1));
};

// Implements [1] Algorithm 3 "PReduction"
//
// Given a covariance matrix `w_x` and real-valued least-squares estimate
// `x_hat`, this algorithm comouted a unimodual matrix Z and the L^T*D*L
// factorization `w_z = z^T * w_z * z = L^T*D*L, which is obtained from the
// L^T*D*L factorization of w_z by updating.
//
// This algorithm also computes z_hat = z^T * x_hat.

template <typename DerivedW, typename DerivedXin, typename DerivedL,
          typename DerivedD, typename DerivedZ, typename DerivedXout> 
bool PReduction(const Eigen::MatrixBase<DerivedW> &w_x,
                const Eigen::MatrixBase<DerivedXin> &x_ls,
                Eigen::MatrixBase<DerivedL> const &l_out,
                Eigen::MatrixBase<DerivedD> const &d_out,
                Eigen::MatrixBase<DerivedZ> const &z_out,
                Eigen::MatrixBase<DerivedXout> const &x_hat_out)
{
    std::cout << "Starting PReduction" << std::endl;
    using Scalar = typename DerivedW::Scalar;
    
    Eigen::MatrixBase<DerivedL> &l = const_cast<Eigen::MatrixBase<DerivedL>&>(l_out);
    Eigen::MatrixBase<DerivedZ> &z = const_cast<Eigen::MatrixBase<DerivedZ>&>(z_out);
    Eigen::MatrixBase<DerivedXout> &x_hat = const_cast<Eigen::MatrixBase<DerivedXout>&>(x_hat_out);
    Eigen::MatrixBase<DerivedD> &d = const_cast<Eigen::MatrixBase<DerivedD>&>(d_out);

    LtdlMinPivot<Eigen::MatrixXd> ltdl;
    if (!ltdl.ComputeDecomposition(w_x, x_ls))
        return false;

    std::cout << "Decomposition computed" << std::endl;

    l = ltdl.matrixL();
    d.noalias() = ltdl.vectorD();
    x_hat.noalias() = ltdl.PermutedVectorA();
    z.noalias() = ltdl.ComputePermutationMatrix();

    const int n = l.rows();
    int k = n-2;

    while (k >= 0)
    {
        std::cout << "k = " << k << std::endl;

        const int kp1 = k+1;
        Scalar l_kp1_k = l.coeff(kp1, k);

        const long mu = std::lround(l_kp1_k);
        const Scalar scalar_mu = static_cast<Scalar>(mu);
        if (mu != 0)
        {
            l_kp1_k -= scalar_mu * l.coeff(kp1, kp1);
        }

        const Scalar delta = d.coeff(k) + l_kp1_k * l_kp1_k * d.coeff(kp1);

        if (delta < d.coeff(kp1))
        {
            // Perform size reductions, if required.
            if (mu != 0)
            {
                for (int i = k+1; i < n; ++i) 
                {
                    IntegerGaussTransform(l, k+1, k, x_hat, z);
                }
            }
            
            // Perform permutations.
            Permute(l, d, k, delta, x_hat, z);
            
            if (k < (n-2))
                k++;
        }
        else
        {
            k--;
        }
    }

    return true;
}
                

































































#endif  // BORNO_PREDUCTION_H__
