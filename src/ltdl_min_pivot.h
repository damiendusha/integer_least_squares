// References:
// [1] Chang, X. et. al., "MLAMBDA: a modified LAMBDA method for integer least-
//     squares estimation", Journal of Geodesy, 79:552-565, 2005.

#ifndef LDLT_MIN_PIVOT_H_
#define LDLT_MIN_PIVOT_H_


#include <eigen3/Eigen/Core>
#include <iostream>

// Solves the problem:
//   P^T * Q * P = L^T * D * L
template <typename MatrixType>
class LtdlMinPivot
{
  public:
    static constexpr auto RowsAtCompileTime = MatrixType::RowsAtCompileTime;
    static constexpr auto ColsAtCompileTime = MatrixType::ColsAtCompileTime;
    static constexpr auto MaxRowsAtCompileTime = MatrixType::MaxRowsAtCompileTime;
    static constexpr auto MaxColsAtCompileTime = MatrixType::MaxColsAtCompileTime;
    using Scalar = typename MatrixType::Scalar;
    // using TranspositionType = typename Eigen::Transpositions<RowsAtCompileTime, 
    //         MaxRowsAtCompileTime>;
    using VectorType = typename Eigen::Matrix<Scalar, RowsAtCompileTime,
        1, 0, MaxRowsAtCompileTime, 1>;

    using PermutationMatrixType = typename Eigen::Matrix<int, RowsAtCompileTime,
        ColsAtCompileTime, 0, MaxRowsAtCompileTime, MaxColsAtCompileTime>;
    
    LtdlMinPivot()
    {
    }

    template <typename InputMatrix, typename InputVector>
    bool ComputeDecomposition(const Eigen::MatrixBase<InputMatrix>& w,
                              const Eigen::MatrixBase<InputVector>& a);

    /// \brief Returns the coefficients of diagonal matrix D.
    Eigen::Diagonal<const MatrixType> vectorD() const
    {
      return matrix_.diagonal();
    }

    /// \brief Returns a view of the strictly lower triangular matrix L.
    Eigen::TriangularView<const MatrixType, Eigen::UnitLower> matrixL() const
    {
        return matrix_.template triangularView<Eigen::UnitLower>();
    }
    
    const VectorType& PermutedVectorA() const
    {
        return a_;
    }

    PermutationMatrixType ComputePermutationMatrix() const
    {
        const int n = permutations_.rows();
        PermutationMatrixType p = PermutationMatrixType::Zero(n ,n);
        for (int j = 0 ; j < n; ++j)
        {
            p.coeffRef(j, permutations_(j)) = 1;
        }

        return p;
    }

  private:
    // Has the decomposition been initialized?
    bool is_initialized_ = false;

    // Used to store the decomposition.
    // - The strictly upper triangle is used as computation scratch space.
    // - The strictly lower triangle corresponds to the coefficents of L.
    // - The diagonal corresponds to the elements of D.
    MatrixType matrix_;

    // Permuted linlear solution.
    VectorType a_;

    using PermStorageType = typename Eigen::Matrix<int, RowsAtCompileTime,
        1, 0, MaxRowsAtCompileTime, 1>;
    PermStorageType permutations_;
    
    VectorType scratch_;
};

template <typename MatrixType>
template <typename InputMatrix, typename InputVector>
bool LtdlMinPivot<MatrixType>::ComputeDecomposition(
    const Eigen::MatrixBase<InputMatrix>& w,
    const Eigen::MatrixBase<InputVector>& a)
{
    const int n = w.rows();
    if (n <= 0)
        return false;
    
    scratch_.resize(n);

    std::cout << "W = " << std::endl << w << std::endl;
    std::cout << "a = " << std::endl << a << std::endl;

    // Store the arguments.
    matrix_ = w;
    a_ = a;

    // Set the pivot to the identity matrix, [0, 1, ..., (n-1)]
    permutations_ = PermStorageType::LinSpaced(n, 0, n-1);

    for (int k = n-1; k >= 0; k--)
    {
        // Select the minimum diagonal value.
        int q;
        const Scalar min_value = matrix_.diagonal().head(k+1).minCoeff(&q);
        std::cout << "q = " << q << std::endl;

        // Error if not PD (note zero comparison);
        if (min_value <= 0)
            return false;

        // No need to swap if already the smallest value.
        if (k != q)
        {
            // Swap the vector elements.
            std::swap(a_.coeffRef(k), a_.coeffRef(q));

            // Swap the permutation rows.
            std::swap(permutations_.coeffRef(k), permutations_.coeffRef(q));

            // Swap rows Q and K of the matrix.
            matrix_.row(k).head(k+1).swap(matrix_.row(q).head(k+1));

            // Swap columns Q and K of the matrix.
            matrix_.col(k).tail(k+1).swap(matrix_.col(q).tail(k+1));
        }
    
        std::cout << "q = " << q << std::endl;

        if (k > 0)
        {
            std::cout << "k = " << k << std::endl;

            // W(k,1:k-1) = W(k,1:k-1)/W(k,k);
            const Scalar d = matrix_.coeff(k, k);
            matrix_.row(k).head(k) *= 1 / d;
       
            std::cout << "k = " << k << std::endl;
            
            // W(1:k-1,1:k-1) = W(1:k-1,1:k-1) - W(k,1:k-1)'*W(k,k)*W(k,1:k-1);
            // TODO: Only compute for the lower triangular portion.
            scratch_.head(k).noalias() = matrix_.row(k).head(k).transpose();
            std::cout << "k = " << k << std::endl;
            matrix_.topLeftCorner(k, k).template selfadjointView<Eigen::Lower>().rankUpdate(
                scratch_.head(k), -d);
            std::cout << "k = " << k << std::endl;
        }
        
        std::cout << "q = " << q << std::endl;
    }
    
    return true;
}
        

#endif  // LDLT_MIN_PIVOT_H_
