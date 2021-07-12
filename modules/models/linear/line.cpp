//
// Created by yiak on 2021/7/12.
//

#include "line.h"

namespace svso {
namespace models {
namespace linear {

void LinearModel::fit(const Eigen::MatrixXd& X, const Eigen::VectorXd& y) {
    // using Eigen to solve least square equation system
    long rows = X.rows();
    long cols = X.cols();

    // normalize the data
    if (normalize_) {
        offsetX_ = X.colwise().mean();
        offsetY_ = y.mean();
    } else {
        offsetX_ = Eigen::VectorXd::Zero(cols);
        offsetY_ = 0.0;
    }

    Eigen::MatrixXd A(rows, cols+1);
    A.block(0,0,rows,cols) = X.rowwise() - offsetX_.transpose();
    A.col(cols).setOnes();

    const Eigen::VectorXd& b = y.array() - offsetY_;

    // using SVD for highest precision
    Eigen::VectorXd beta = A.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(b);

    // extract parameters and produce residual statistical report
    calldata_.coefficients = beta.segment(0, cols);
    calldata_.intercept = beta[cols];
    is_fitted_ = true;

    Eigen::VectorXd fittedVals;
    Eigen::VectorXd residuals;
    predict(X, fittedVals);
    residuals = y - fittedVals;

    std::vector<double> residuals_array(residuals.data(), residuals.data() + rows);
    ResStat stat(residuals_array);
    stat_ = std::move(stat);
}

    } // linear
  } // models
} // svso