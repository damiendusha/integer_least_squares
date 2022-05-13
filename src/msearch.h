#ifndef M_SEARCH_H__
#define M_SEARCH_H__

// References:
// [1] Chang, X. et. al., "MLAMBDA: a modified LAMBDA method for integer least-
//     squares estimation", Journal of Geodesy, 79:552-565, 2005.

#include <eigen3/Eigen/Core>

#include <limits>

template <typename IntType, typename Scalar>
IntType Sign(const Scalar x)
{
    constexpr IntType plus_one = 1;
    constexpr IntType minus_one = -1;

    return x <= 0 ? minus_one : plus_one;
}

template <typename DerivedIn, typename DerivedOut>
void ComputeSignVector(const Eigen::MatrixBase<DerivedIn> &in,
                       Eigen::MatrixBase<DerivedOut> const &out)
{
    Eigen::MatrixBase<DerivedOut>& o = 
            const_cast<Eigen::MatrixBase<DerivedOut>&>(out);

    o.derived().resize(in.rows(), in.cols());
    for (int col = 0; col < in.cols(); ++col) {
        for (int row = 0 ; row < in.rows(); ++row)
        {
            // The paper defines sign(x) as -1 when x <= 0. We therefore
            // do not use std::copysign for the comparison.
            // o.coeff(row, col) = std::copysign(one, in.coeff(row, col));
            o.coeff(row, col) = Sign(in.coeff(row, col));
        }
    }
}
                       
// [1] Algorithm 3.3.
template <typename DerivedL, typename DerivedD, typename DerivedZ,
          typename DerivedO, typename DerivedR>
bool MSearch(const Eigen::MatrixBase<DerivedL> &l,
             const Eigen::MatrixBase<DerivedD> &d,
             const Eigen::MatrixBase<DerivedZ> &z_hat,
             const int p,
             Eigen::MatrixBase<DerivedO> const &opt_out,
             Eigen::MatrixBase<DerivedR> const &resid_out,
             const int max_iter = 10000)
{
    using Scalar = typename DerivedL::Scalar;
    using ScalarVector = typename Eigen::MatrixBase<DerivedZ>::PlainObject;

    using IntType = typename DerivedO::Scalar;
    using IntVector = Eigen::VectorXi;

    using Index = typename DerivedL::Index;
    
    Eigen::MatrixBase<DerivedO> &opt = 
            const_cast<Eigen::MatrixBase<DerivedO>&>(opt_out);
    Eigen::MatrixBase<DerivedR> &resid =
            const_cast<Eigen::MatrixBase<DerivedR>&>(resid_out);

    // Current chi^x search value.
    Scalar max_dist = std::numeric_limits<Scalar>::max();
    
    resid.derived().resize(p, 1);
    resid.setZero();

    const int n = l.rows();
    int k = n-1;
    
    ScalarVector dist = ScalarVector::Zero(n, 1);
    
    bool end_search = false;
    int count = 0;
    
    Eigen::MatrixXd s = Eigen::MatrixXd::Zero(n, n);
       
    // See [1] Eq.(17).
    ScalarVector z_bar = ScalarVector::Zero(n, 1);
    z_bar.coeffRef(k) = z_hat(k);
    
    IntVector z = IntVector::Zero(n, 1);
    z.coeffRef(k) = std::lround(z_bar.coeff(k));
    
    ScalarVector y = ScalarVector::Zero(n, 1);
    y.coeffRef(k) = z_bar.coeff(k) - z.coeff(k);
    
    IntVector step = IntVector::Zero(n, 1);
    step.coeffRef(k) = Sign<IntType>(y.coeff(k));
    
    Index imax = 0;

    int iter = 0;
    while (!end_search && iter < max_iter)
    {
        iter++;
        std::cout << "Iteration = " << iter << std::endl;
        std::cout << "k = " << k << std::endl;
        
        const Scalar new_dist = dist(k) + y(k) * y(k) / d(k);
        if (new_dist < max_dist)
        {
            // Case 1: Move down.
            if (k != 0)
            {
                std::cout << "Case 1" << std::endl;
                k--;
                dist(k) = new_dist;
                s.row(k).head(k+1).noalias() = s.row(k+1).head(k+1)
                        + (z(k+1) - z_bar(k+1)) * l.row(k+1).head(k+1);
                        
                // See Eq.(17).
                z_bar(k) = z_hat(k) + s(k,k);
                z(k) = std::lround(z_bar(k));
                y(k) = z_bar(k) - z(k);
                step(k) = Sign<IntType>(y(k));
            }
            // Case 2: Store the found candidate.
            else
            {
                std::cout << "Case 2" << std::endl;
                if (count < p)
                {
                    // RTKLIB does this a little differently to the published
                    // algorithm. We adopt their trick here.
                    if (count == 0 || new_dist > resid(imax))
                        imax = count;

                    // Store the first p initial points.
                    opt.col(count) = z.template cast<IntType>();
                    resid(count) = new_dist;
                    count++;
                }
                else
                {
                    // RTKLIB does this a little differently to the published
                    // algorithm. We adopt their trick here.
                    if (new_dist < resid(imax))
                    {
                        opt.col(imax) = z.template cast<IntType>();
                        resid.coeffRef(imax) = new_dist;
                        resid.maxCoeff(&imax);
                    }
                    max_dist = resid.coeff(imax);
                }
                
                z(0) += step(0);
                y(0) = z_bar(0) - z(0);
                step(0) = -step(0) - Sign<IntType>(step(0));
            }
        }
        // Case 3: Exit or move up.
        else
        {
            std::cout << "Case 3" << std::endl;
            if (k >= (n-1))
            {
                end_search = true;
            }
            else
            {
                k++;

                // Next valid integer.
                z(k) += step(k);
                y(k) = z_bar(k) - z(k);
                step(k) = -step(k) - Sign<IntType>(step(k));
            }
        }
    }

    return iter < max_iter;
}

#endif  // M_SEARCH_H__
