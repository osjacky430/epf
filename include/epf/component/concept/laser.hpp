#ifndef LASER_CONCEPT_HPP_
#define LASER_CONCEPT_HPP_

#include "epf/util/traits.hpp"
#include <boost/concept/usage.hpp>
#include <vector>

namespace epf {

template <typename LaserData, typename Tag = void>
auto range_data(LaserData const&);

template <typename LaserData, typename Tag = void>
auto bearing_data(LaserData const&);

template <typename LaserData>
struct LaserDataConcept {  // NOLINT(*-special-member-functions), because this class is not meant to be
                           // instantiated
  BOOST_CONCEPT_USAGE(LaserDataConcept) {
    [[maybe_unused]] auto const laser_range   = range_data(LaserData{});
    [[maybe_unused]] auto const laser_bearing = bearing_data(LaserData{});

    [[maybe_unused]] auto range_begin = laser_range.begin();
    [[maybe_unused]] auto range_end   = laser_range.end();

    [[maybe_unused]] auto bearing_begin = laser_bearing.begin();
    [[maybe_unused]] auto bearing_end   = laser_bearing.end();
  }
};

}  // namespace epf

#endif