#ifndef EPF_EXCEPTIONS_HPP_
#define EPF_EXCEPTIONS_HPP_

#include <stdexcept>

namespace epf {

struct NotImplementedError : std::runtime_error {
  using std::runtime_error::runtime_error;
};

}  // namespace epf

#endif