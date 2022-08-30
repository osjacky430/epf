#ifndef ENUM_HPP_
#define ENUM_HPP_

namespace epf {

enum class MeasurementResult { NoMeasurement, Estimated };

enum class Prediction { NoUpdate, Updated };

enum class SamplingResult { NoSampling, Sampled };

}  // namespace epf

#endif