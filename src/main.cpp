#include "ltdl_min_pivot.h"
#include "preduction.h"
#include "msearch.h"

#include <eigen3/Eigen/Dense>
#include <iostream>

int main(void)
{
    int m = 5;
    int n = 3;
    Eigen::MatrixXd a;
    a.resize(m, n);
    a << 
        0.0241027,   1.5613544,  -0.1136238,
       -1.3428477,  -0.0072077,   0.5468784,
        0.9764972,   0.8415361,   0.3113262,
        1.1045219,   1.0146555,  -0.2640554,
        1.9128498,   0.7557230,   0.1055820;

    Eigen::Vector3d x_true(1, -2, 3);

    Eigen::Matrix<double, 5, 1> y;
    y << 2.1, -1.6, 1.3, 0.6, 3.1;
    y *= 1e-3;
    y += a * x_true;

    Eigen::VectorXd x_in = a.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(y);
    Eigen::MatrixXd w = (a.transpose() *a).inverse();
    
    Eigen::MatrixXd l;
    Eigen::VectorXd d;
    Eigen::MatrixXi z;
    Eigen::VectorXd x_out;
    PReduction(w, x_in, l, d, z, x_out);

    std::cout << "l = " << std::endl << l << std::endl;
    std::cout << "d = " << std::endl << d << std::endl;
    std::cout << "z = " << std::endl << z << std::endl;
    std::cout << "x_out = " << std::endl << x_out << std::endl;

    int p = 2;
    
    Eigen::MatrixXi opt(n, p);
    Eigen::Vector2d resid;
    MSearch(l,d,x_out,p, opt, resid);

    std::cout << "Opt = " << std::endl << opt << std::endl;
    std::cout << "Resid = " << std::endl << resid << std::endl;
    
    Eigen::MatrixXi out = z.transpose() * opt;
    std::cout << "Out = " << std::endl << out << std::endl;

    return 0;
}
