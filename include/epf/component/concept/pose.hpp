#ifndef POSE_CONCEPT_HPP_
#define POSE_CONCEPT_HPP_

#include "epf/core/state.hpp"
#include <boost/concept/usage.hpp>

namespace epf {

// optional concept?! sfinae?
// temporary, not sure if this is the best api
template <typename State>
struct Pose2DConcept {  // NOLINT(*-special-member-functions), because this class is not meant to be
                        // instantiated
  BOOST_CONCEPT_USAGE(Pose2DConcept) {
    using StateValue  = typename StateTraits<State>::ValueType;
    using StateVector = typename StateTraits<State>::ArithmeticType;

    StateVector s{};
    [[maybe_unused]] StateValue& ref_x = x_coor<State>(s);
    [[maybe_unused]] StateValue& ref_y = y_coor<State>(s);
    [[maybe_unused]] StateValue& ret_w = w_coor<State>(s);
  }
};

}  // namespace epf

#endif