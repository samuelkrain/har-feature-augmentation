#ifndef HAR_MODEL_SETTINGS_H_
#define HAR_MODEL_SETTINGS_H_

const int kSegmentSize = 128;
const int kNumChannels = 6;
constexpr int kInputElementCount = kSegmentSize*kNumChannels;
const int kFeatureElementCount = 40;
constexpr int kInputSampleSize = kInputElementCount+kFeatureElementCount+1;

#endif