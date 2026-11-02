/*
 * Copyright 2016, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#pragma once

#include <stdexcept>
#include <cmath>
#include <filesystem>
#include <cctag/utils/Statistics.hpp>
#include "TestLog.h"


enum class CudaMode { FromParameters, Disabled, Enabled };

class TestRunner
{
  const std::filesystem::path _inputDirPath;
  const std::filesystem::path _outputDirPath;
  const CudaMode _useCuda;
  std::vector<std::filesystem::path> _inputFilePaths;
  
  void adjustParameters(cctag::Parameters& parameters);
  
public:
  TestRunner(const std::string& inputDir, const std::string& outputDir, CudaMode useCuda);
  void generateReferenceResults(cctag::Parameters parameters);
  void generateTestResults();
};

class TestChecker
{
  const std::filesystem::path _referenceDirPath;
  const std::filesystem::path _testDirPath;
  const float _epsilon;
  
  using PathVector = std::vector<std::filesystem::path>;
  
  struct check_error : public std::runtime_error
  {
    explicit check_error(const std::string& what) : runtime_error(what)
    { }
  };

  PathVector _referenceFilePaths;
  PathVector _testFilePaths;
  cctag::numerical::RunningMeanVariance<false> _elapsedDiffAcc;  // over all frames in the dataset
  cctag::numerical::RunningMeanVariance<false> _qualityDiffAcc;  // over all tags in the dataset
  bool _failed;
  
  void check(const std::filesystem::path& testFilePath);
  void compare(FileLog& referenceLog, FileLog& testLog);
  void compare(FrameLog& referenceLog, FrameLog& testLog, size_t frame);
  void compare(const DetectedTag& referenceTag, const DetectedTag& testTag, size_t frame);
  std::filesystem::path testToReferencePath(const std::filesystem::path& testPath);
  
public:
  TestChecker(const std::string& referenceDir, const std::string& testDir, float epsilon);
  bool check();
  float elapsedTimeDifferenceMean() { return _elapsedDiffAcc.mean(); }
  float elapsedTimeDifferenceStdev() { return std::sqrt(_elapsedDiffAcc.variance()); }
  float qualityDifferenceMean() { return _qualityDiffAcc.mean(); }
  float qualityDifferenceStdev() { return std::sqrt(_qualityDiffAcc.variance()); }
};
