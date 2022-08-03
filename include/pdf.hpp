#ifndef PDF_HPP_
#define PDF_HPP_

#include <array>

namespace particle_filter {

// 1. How to represent a PDF?
// 2. Should we consider extensibilty for different PDF (Yes, probably)
class ProbabilityDensityFunction {
  // mean, covariance, consider using Eigen

  /* col_vector<3> mean */
  /* square_matrix<3> covariance */

  // cache determinant as we may use it pretty often
  // double determinant_;

  // cache rotational and translation covariance
  // square_matrix<3> rotation_cov_;
  // square_matrix<3> translation_cov_;
 public:
  /* mean return type */ void sample() noexcept;
};

}  // namespace particle_filter

#endif