/*
 * Copyright 2016, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#include <cctag/Identification.hpp>
#include <cctag/ImageCut.hpp>
#include <cctag/optimization/conditioner.hpp>
#include <cctag/geometry/2DTransform.hpp>
#include <cctag/utils/Defines.hpp>

#include <cctag/geometry/Circle.hpp>
#include <cctag/utils/Talk.hpp>

#ifdef CCTAG_WITH_CUDA
#include "cctag/cuda/tag.h"
#endif

#include <cctag/utils/Statistics.hpp>
#include <array>
#include <cassert>
#include <stdexcept>

#include <cmath>
#include <chrono>
#include <mutex>
#include <vector>

#include <tbb/tbb.h>

#include "cctag/cuda/onoff.h"

namespace cctag {
namespace identification {

template<typename Kernel>
static std::pair<float, float> convImageCutImpl(const Kernel& kernel, ImageCut& cut);

// Append one vote per readable cut; the caller applies the probability threshold.
bool orazioDistanceRobust(std::vector<std::list<float>>& votesById,
                          const RadiusRatioBank& radiusRatios,
                          const std::vector<ImageCut>& cuts,
                          float /* minIdentProba */)
{
    if(cuts.empty() || radiusRatios.empty())
        return false;

    std::mutex votesMutex;
    tbb::parallel_for(std::size_t(0), cuts.size(), [&](std::size_t cutIndex) {
        const ImageCut& cut = cuts[cutIndex];
        if(cut.outOfBounds())
            return;

        const auto& signal = cut.imgSignal();
        constexpr std::size_t ignoredLeadingSamples = 30;
        if(signal.size() <= ignoredLeadingSamples)
            return;

        numerical::RunningMeanVariance<> statistics;
        statistics = std::for_each(signal.begin() + ignoredLeadingSamples, signal.end(), statistics);
        const float signalMean = statistics.mean();
        const float signalVariance = statistics.variance();
        if(!std::isfinite(signalMean) || !std::isfinite(signalVariance) || signalVariance <= 0.f)
            return;

        numerical::RunningMean darkSamples;
        numerical::RunningMean lightSamples;
        bool passedFirstDarkSample = false;
        for(float sample : signal)
        {
            if(sample < signalMean)
                passedFirstDarkSample = true;

            if(passedFirstDarkSample)
            {
                if(sample < signalMean)
                    darkSamples(sample);
                else
                    lightSamples(sample);
            }
        }
        const float lightMean = lightSamples.mean();
        const float darkMean = darkSamples.mean();
        // Both classes are needed; a flat profile can leave one mean undefined.
        if(!std::isfinite(lightMean) || !std::isfinite(darkMean))
            return;
        const float sampleStep = (cut.endSig() - cut.beginSig()) / (signal.size() - 1.f);
        std::vector<float> expectedSignal(signal.size());
        float bestProbability = -1.f;
        std::size_t bestMarkerId = 0;

        for(std::size_t markerId = 0; markerId < radiusRatios.size(); ++markerId)
        {
            float position = cut.beginSig();
            for(float& expectedSample : expectedSignal)
            {
                std::size_t crossedRings = 0;
                for(float ratio : radiusRatios[markerId])
                    if(1.f / ratio <= position)
                        ++crossedRings;

                expectedSample = crossedRings % 2 ? -1.f : 1.f;
                position += sampleStep;
            }

            float distance = 0.f;
            for(std::size_t sampleIndex = 0; sampleIndex < signal.size(); ++sampleIndex)
            {
                distance += dis(signal[sampleIndex], expectedSignal[sampleIndex], darkMean, lightMean, signalVariance);
            }
            // Leading samples are outside the statistics window but still enter the distance.
            if(std::isnan(distance))
                return;
            const float probability = std::exp(-distance);
            // Keep the last marker ID when probabilities tie.
            if(probability >= bestProbability)
            {
                bestProbability = probability;
                bestMarkerId = markerId;
            }
        }

        std::lock_guard<std::mutex> lock(votesMutex);
        votesById[bestMarkerId].push_back(bestProbability);
    });
    return true;
}

void createRectifiedCutImage(const std::vector<ImageCut> & vCuts, cv::Mat & output)
{
  output = cv::Mat(vCuts.size(), vCuts.front().imgSignal().size(), CV_8UC1);
  for(int i=0 ; i<vCuts.size() ; ++i)
  {
    const ImageCut & cut = vCuts[i];
    for(int j=0 ; j < cut.imgSignal().size() ; ++j)
    {
      output.at<uchar>(i,j) = (uchar) cut.imgSignal()[j];
    }
  }
}

/**
 * @brief Extract a rectified 1D signal along an image cut based on an homography.
 *
 * @param[out] cut image cut holding the rectified image signal
 * @param[in] src source grayscale image (uchar)
 * @param[in] mHomography cctag->image homography
 * @param[in] mInvHomography image->cctag homography
 */
void extractSignalUsingHomography(
        cctag::ImageCut & cut,
        const cv::Mat & src,
        const Eigen::Matrix3f & mHomography,
        const Eigen::Matrix3f & mInvHomography)
{
  using namespace cctag::numerical;
  // Bounds belong to this sampling pass, not an earlier centre candidate.
  cut.setOutOfBounds(false);
  const std::size_t nSamples = cut.imgSignal().size();
  if(nSamples < 2)
  {
    cut.setOutOfBounds(true);
    return;
  }

  float xStart, xStop, yStart, yStop;
  
  float backProjStopX, backProjStopY;
  applyHomography(backProjStopX, backProjStopY, mInvHomography, cut.stop().x(), cut.stop().y());
  
  // Check whether the signal to be collected start at 0.f and stop at 1.f
  if ( cut.beginSig() != 0.f)
  {
    xStart = backProjStopX * cut.beginSig();
    yStart = backProjStopY * cut.beginSig();
  }else
  {
    xStart = 0;
    yStart = 0;
  }
  if ( cut.endSig() != 1.f)
  {
    xStop = backProjStopX * cut.endSig();
    yStop = backProjStopY * cut.endSig();
  }else
  {
    xStop  = backProjStopX;
    yStop  = backProjStopY;
  }

  // Compute the steps stepX and stepY along x and y.
  const float stepX = ( xStop - xStart ) / ( nSamples - 1.f );
  const float stepY = ( yStop - yStart ) / ( nSamples - 1.f );

  float xRes, yRes;

  float x =  xStart;
  float y =  yStart;
  
  for( std::size_t i = 0; i < nSamples; ++i )
  {
    applyHomography(xRes, yRes, mHomography, x, y);

    if ( xRes >= 0.f && xRes < src.cols-1 &&
         yRes >= 0.f && yRes < src.rows-1 )
    {
      // Bilinear interpolation
      cut.imgSignal()[i] = getPixelBilinear( src, xRes, yRes);
    }
    else
    {
      cut.setOutOfBounds(true);
    }
    
    x += stepX;
    y += stepY;
  }
  //const float sigma = 1.f;
  //blurImageCut(sigma, cut);
}

void blurImageCut(float /* sigma */, std::vector<float>& signal)
{
    static constexpr std::array<float, 9> kernel{
      {0.0276f, 0.0663f, 0.1238f, 0.1802f, 0.2042f, 0.1802f, 0.1238f, 0.0663f, 0.0276f}};
    std::vector<float> output(signal.size());
    constexpr std::ssize_t halfWidth = (kernel.size() - 1) / 2;
    for(std::size_t sampleIndex = 0; sampleIndex < signal.size(); ++sampleIndex)
    {
        float smoothedSample = 0.f;
        for(std::size_t kernelIndex = 0; kernelIndex < kernel.size(); ++kernelIndex)
        {
            const auto sourceIndex = std::clamp<std::ssize_t>(static_cast<std::ssize_t>(sampleIndex) - halfWidth +
                                                                static_cast<std::ssize_t>(kernelIndex),
                                                              0,
                                                              signal.size() - 1);
            smoothedSample += signal[sourceIndex] * kernel[kernelIndex];
        }
        output[sampleIndex] = smoothedSample;
    }
    signal.swap(output);
}

void blurImageCut(float sigma, ImageCut& cut) { blurImageCut(sigma, cut.imgSignal()); }

void extractSignalUsingHomographyDeprec(
        cctag::ImageCut & rectifiedCut,
        const cv::Mat & src,
        Eigen::Matrix3f & mHomography,
        std::size_t nSamples,
        float begin,
        float end)
{
  using namespace cctag::numerical;

  assert( rectifiedCut.imgSignal().size() == 0 );
  assert( end >= begin );
  
  // Check wheter the image signal size has been properly allocated.
  assert( nSamples == rectifiedCut.imgSignal().size() );

  nSamples = rectifiedCut.imgSignal().size();
  
  const float stepXi = ( end - begin ) / ( nSamples - 1.f );

  rectifiedCut.start() = getHPoint( begin, 0.f, mHomography );
  rectifiedCut.stop() = cctag::DirectedPoint2d<Eigen::Vector3f>( getHPoint( end, 0.f, mHomography ), 0.f, 0.f);

  std::vector<std::size_t> idxNotInBounds;
  for( std::size_t i = 0; i < nSamples; ++i )
  {
    const float xi = i * stepXi + begin;
    const cctag::Point2d<Eigen::Vector3f> hp = getHPoint( xi, 0.f, mHomography );

    if ( hp.x() >= 1.f && hp.x() <= src.cols-1 &&
         hp.y() >= 1.f && hp.y() <= src.rows-1 )
    {
      // Bilinear interpolation
      rectifiedCut.imgSignal()[i] = getPixelBilinear( src, hp.x(), hp.y());
    }
    else
    {
      rectifiedCut.setOutOfBounds(true);
    }
  }
}

/// Code usable to apply a 1D gaussian filtering at the end of the extractSignalUsingHomography
///
//  float guassOneD[] = { 0.0044, 0.0540, 0.2420, 0.3991, 0.2420, 0.0540, 0.0044 };
//  for( std::size_t i = 0; i < nSamples; ++i )
//  {
//    float tmp = 0;
//    for ( std::size_t j=0 ; j<7; ++j)
//    {
//      if ( (i-3+j >= 0) && (i-3+j < nSamples) )
//      {
//        tmp += rectifiedCut.imgSignal()(i-3+j)*guassOneD[j];
//      }
//    }
//    rectifiedCut.imgSignal()(i) = tmp;
//  }
///


/**
 * @brief Extract a regularly sampled 1D signal along an image cut.
 * 
 * @param[out] cut image cut that will hold the 1D image signal regularly 
 *             collected from cut.beginSig() to cut.endSig()
 * @param[in] src source gray scale image (uchar)
 */
void cutInterpolated(
        cctag::ImageCut & cut,
        const cv::Mat & src)
{
    cut.setOutOfBounds(false);
    const std::size_t nSamples = cut.imgSignal().size();
    if(nSamples < 2)
    {
        cut.setOutOfBounds(true);
        return;
    }
    float xStart, yStart, xStop, yStop;
    const float diffX = cut.stop().x() - cut.start().x();
    const float diffY = cut.stop().y() - cut.start().y();

    // Check whether the signal to be collected start at 0.f and stop at 1.f
    if(cut.beginSig() != 0.f)
    {
        // Don't start at the beginning of the cut.
        xStart = cut.start().x() + diffX * cut.beginSig();
        yStart = cut.start().y() + diffY * cut.beginSig();
    }
    else
    {
        xStart = cut.start().x();
        yStart = cut.start().y();
  }
  if ( cut.endSig() != 1.f)
  {
    // Don't stop at the end of the cut.
    xStop = cut.start().x() + diffX * cut.endSig();
    yStop = cut.start().y() + diffY * cut.endSig();
  }else
  {
    xStop = cut.stop().x();
    yStop = cut.stop().y();
  }
  
  // Compute the steps stepX and stepY along x and y.
  // We will work on the image while assuming an affine transformation.
  // Therefore, steps are computed in the pixel plane, regularly spaced from
  // the outer ellipse's center (cut.start()) to an outer point (cut.stop()).
  const float stepX = ( xStop - xStart ) / ( nSamples - 1.f );
  const float stepY = ( yStop - yStart ) / ( nSamples - 1.f );

  float x =  xStart;
  float y =  yStart;
  
  for( std::size_t i = 0; i < nSamples; ++i )
  {
    if ( x >= 1.f && x < src.cols-1 &&
         y >= 1.f && y < src.rows-1 )
    {
      //CCTAG_COUT_VAR2(x,y);
      // put pixel value to rectified signal
      cut.imgSignal()[i] = float(getPixelBilinear( src, x, y));
    }
    else
    {
      // push black
      cut.setOutOfBounds(true);
      break;
    }
    // Modify x and y to the next element.
    //CCTAG_COUT_VAR2(x,y);
    //CCTAG_COUT_VAR2(stepX,stepY);
    x += stepX;
    y += stepY;
    //CCTAG_COUT_VAR2(x,y);
  }
}

/**
 * @brief Collect signals (image cuts) from center to outer ellipse points
 * 
 * @param[out] cuts collected cuts
 * @param[in] src source gray scale image (uchar)
 * @param[in] center outer ellipse center
 * @param[in] outerPoints outer ellipse points
 * @param[in] nSamplesInCut number of samples collected in an image cut
 * @param[in] beginSig offset from which the signal must be collected (in [0 1])
 */
void collectCuts(
        std::vector<cctag::ImageCut> & cuts,
        const cv::Mat & src,
        const cctag::Point2d<Eigen::Vector3f> & center,
        const std::vector< cctag::DirectedPoint2d<Eigen::Vector3f> > & outerPoints,
        std::size_t nSamplesInCut,
        float beginSig )
{
  // Collect all the 1D image signals from center to the outer points.
  cuts.reserve( outerPoints.size() );
  for( const cctag::DirectedPoint2d<Eigen::Vector3f> & outerPoint : outerPoints )
  {
    // Here only beginSig is set based on the input argument beginSig while endSig is set to 1.f as 
    // any type of cctags encodes, by construction, a 1D bar-code until the outer ellipse (image 
    // of the unit circle).
    cuts.emplace_back(center, outerPoint, beginSig, 1.f, nSamplesInCut );
    cctag::ImageCut & cut = cuts.back();
    cutInterpolated( cut, src);
    // Remove the cut from the vector if out of the image bounds.
    if ( cut.outOfBounds() )
    {
      CCTAG_COUT_VAR_OPTIM(cut.outOfBounds());
      cuts.pop_back();
    }
  }
}

/**
 * @brief Select a subset of image cuts appropriate for the image center optimisation.
 * This selection aims at "maximizing" the variance of the image signal over all the 
 * selected cuts while ensuring a "good" distribution of the selected outer points
 * around the imaged center.
 *
 * @param[out] vSelectedCuts selected image cuts
 * @param[in] selectSize number of desired cuts to select
 * @param[in] collectedCuts all the collected cuts
 * @param[in] src source gray scale image (uchar)
 */
void selectCutCheapUniform( std::vector< cctag::ImageCut > & vSelectedCuts,
        std::size_t selectSize,
        const cctag::numerical::geometry::Ellipse & outerEllipse,
        std::vector<cctag::ImageCut> & collectedCuts,
        const cv::Mat & src,
        const float scale,
        const size_t numSamplesOuterEdgePointsRefinement)
{
  using namespace cctag::numerical;

  selectSize = std::min( selectSize, collectedCuts.size() );

  std::vector<float> varCuts;
  varCuts.reserve(collectedCuts.size());
  for( const cctag::ImageCut & cut : collectedCuts )
  {
    RunningMeanVariance<> acc;
    acc = std::for_each( cut.imgSignal().begin(), cut.imgSignal().end(), acc );
    varCuts.push_back( acc.variance() );
  }
  
  const float varMax = *std::max_element(varCuts.begin(),varCuts.end());
  
  vSelectedCuts.clear();
  vSelectedCuts.reserve(selectSize);

  // Initialize vector of indices of sharp cuts
  std::vector<std::size_t> indToAdd;
  indToAdd.reserve(varCuts.size());
  
  for(std::size_t iCut = 0 ; iCut < varCuts.size() ; ++iCut)
  {
    ImageCut & cut = collectedCuts[iCut];
    cut.stop() = DirectedPoint2d<Eigen::Vector3f>(pointOnEllipse( outerEllipse, cut.stop() ), cut.stop().dX(), cut.stop().dY() );
    if ( outerEdgeRefinement(cut, src, scale, numSamplesOuterEdgePointsRefinement) )
      if ( varCuts[iCut]/varMax > 0.5f )
        indToAdd.push_back(iCut);
  }
  
  const float step = std::max(1.f, (float) indToAdd.size() / (float) ( selectSize ));
  
  for(std::size_t k=0 ; ; ++k)
  {
    if ( ( std::size_t(k*step) < indToAdd.size() ) && ( vSelectedCuts.size() < selectSize) )
    {
      ImageCut & cut = collectedCuts[indToAdd[std::size_t(k*step)]];
      vSelectedCuts.push_back( cut );
    }else{
      break;
    }
  }
}

bool outerEdgeRefinement(ImageCut& cut, const cv::Mat& src, float scale, std::size_t sampleCount)
{
    const float refinementLength = 3.f * std::sqrt(2.f) * scale;
    const float halfWidth = refinementLength / 2.f;
    const Eigen::Vector2f gradientDirection = cut.stop().gradient() / cut.stop().gradient().norm();
    const auto originalStop = cut.stop();
    const Eigen::Vector2f halfOffset = halfWidth * gradientDirection;
    const Point2d<Eigen::Vector3f> start(originalStop.x() - halfOffset.x(), originalStop.y() - halfOffset.y());
    const DirectedPoint2d<Eigen::Vector3f> stop(originalStop.x() + halfWidth * gradientDirection.x(),
                                                originalStop.y() + halfWidth * gradientDirection.y(),
                                                originalStop.dX(),
                                                originalStop.dY());

    ImageCut edgeCut(start, stop, sampleCount);
    cutInterpolated(edgeCut, src);
    if(edgeCut.outOfBounds())
        return false;

    static constexpr std::array<std::array<float, 9>, 3> derivativeKernels{{
      {-0.0000f, -0.0003f, -0.1065f, -0.7863f, 0.f, 0.7863f, 0.1065f, 0.0003f, 0.0000f}, // sigma = 0.5
      {-0.0044f, -0.0540f, -0.2376f, -0.3450f, 0.f, 0.3450f, 0.2376f, 0.0540f, 0.0044f}, // sigma = 1
      {-0.0366f, -0.1113f, -0.1801f, -0.1594f, 0.f, 0.1594f, 0.1801f, 0.1113f, 0.0366f}  // sigma = 1.5
    }};

    auto strongestPeak = convImageCutImpl(derivativeKernels.front(), edgeCut);
    for(std::size_t kernelIndex = 1; kernelIndex < derivativeKernels.size(); ++kernelIndex)
    {
        const auto peak = convImageCutImpl(derivativeKernels[kernelIndex], edgeCut);
        // Keep the first kernel on ties, as the original map insertion did.
        if(strongestPeak.first < peak.first)
            strongestPeak = peak;
    }

    const float sampleStep = refinementLength / (static_cast<float>(sampleCount) - 1.f);
    const float peakLocation = strongestPeak.second;
    cut.stop() = DirectedPoint2d<Eigen::Vector3f>(start.x() + sampleStep * peakLocation * gradientDirection.x(),
                                                  start.y() + sampleStep * peakLocation * gradientDirection.y(),
                                                  originalStop.dX(),
                                                  originalStop.dY());
    return true;
}

template<typename Kernel>
static std::pair<float,float> convImageCutImpl(const Kernel & kernel, ImageCut & cut)
{
  const auto& signal = cut.imgSignal();
  if(signal.empty())
    throw std::invalid_argument("convImageCut requires a non-empty signal");

  const std::size_t sizeCut = signal.size();
  const std::size_t sizeKernel = kernel.size();
  const std::size_t halfSize = (sizeKernel - 1) / 2;
  float maxValue = 0.f;
  float maxLocation = 0.f;

  for(std::ssize_t i = 0; i < sizeCut; ++i)
  {
    float tmp = 0;
    for(std::size_t j = 0; j < sizeKernel; ++j)
    {
      const std::ssize_t res = i - halfSize + j;
      if(res < 0)
        tmp += signal[0] * kernel[j];
      else if(res >= sizeCut)
        tmp += signal[sizeCut - 1] * kernel[j];
      else
        tmp += signal[i - halfSize + j] * kernel[j];
    }
    // Match max_element: retain the first maximum, even for a first-sample NaN.
    if(i == 0 || maxValue < tmp)
    {
      maxValue = tmp;
      maxLocation = static_cast<float>(i);
    }
  }
  return {maxValue, maxLocation};
}

std::pair<float,float> convImageCut(const std::vector<float> & kernel, ImageCut & cut)
{
  return convImageCutImpl(kernel, cut);
}

/**
 * @brief Collect rectified 1D signals along image cuts.
 *
 * @param[out] vCuts set of the image cuts whose the rectified signal is to be to computed
 * @param[in] mHomography transformation cctag->image used to sample the rectified signal
 * @param[in] src source gray scale image (uchar)
 */
void getSignals(
        std::vector< cctag::ImageCut > & vCuts,
        const Eigen::Matrix3f & mHomography,
        const cv::Mat & src)
{
  Eigen::Matrix3f mInvHomography = mHomography.inverse();
  for( cctag::ImageCut & cut : vCuts )
  {
    extractSignalUsingHomography( cut, src, mHomography, mInvHomography);
  }
}

void computeHomographyFromEllipseAndImagedCenter(const numerical::geometry::Ellipse& ellipse,
                                                 const Point2d<Eigen::Vector3f>& center,
                                                 Eigen::Matrix3f& homography)
{
    Eigen::Matrix3f canonicalEllipse, imageToCanonical, canonicalToImage;
    ellipse.getCanonicForm(canonicalEllipse, imageToCanonical, canonicalToImage);
    float centerX, centerY;
    applyHomography(centerX, centerY, imageToCanonical, center.x(), center.y());
    const float q11 = canonicalEllipse(0, 0);
    const float q22 = canonicalEllipse(1, 1);
    const float q33 = canonicalEllipse(2, 2);

    // Closed-form plane-to-image homography in the ellipse's canonical frame.
    homography(0, 0) = q33;
    homography(1, 0) = 0.f;
    homography(2, 0) = -q11 * centerX;
    homography(0, 1) = q22 * centerX * centerY;
    homography(1, 1) = -q11 * centerX * centerX - q33;
    homography(2, 1) = q22 * centerY;
    homography(0, 2) = -q33 * centerX;
    homography(1, 2) = -q33 * centerY;
    homography(2, 2) = -q33;

    const std::array<float, 3> columnScales{
      {std::sqrt(q22 * q33 / q11 * (q11 * centerX * centerX + q22 * centerY * centerY + q33)),
       q33,
       std::sqrt(-q22 * (q11 * centerX * centerX + q33))}};
    for(int row = 0; row < 3; ++row)
        for(int column = 0; column < 3; ++column)
            homography(row, column) *= columnScales[column];
    homography = canonicalToImage * homography;
}

bool refineConicFamilyGlob(int tagIndex,
                           Eigen::Matrix3f& homography,
                           Point2d<Eigen::Vector3f>& optimalPoint,
                           std::vector<ImageCut>& cuts,
                           const cv::Mat& src,
                           cctag::TagPipe* cudaPipe,
                           const numerical::geometry::Ellipse& outerEllipse,
                           const Parameters& params,
                           cctag::NearbyPoint* nearbyPointBuffer,
                           float& residual)
{
    if(cuts.empty())
        return false;

    auto& visualDebug = CCTagVisualDebug::instance();
    visualDebug.newSession("refineConicPts");
    for(const ImageCut& cut : cuts)
        visualDebug.drawPoint(cut.stop(), color_red);
    visualDebug.newSession("centerOpt");
    visualDebug.drawPoint(optimalPoint, color_green);

#ifdef CCTAG_WITH_CUDA
    if(cudaPipe)
    {
        if(!cudaPipe->imageCenterRetrieve(tagIndex, optimalPoint, residual, homography, params, nearbyPointBuffer))
            return false;
    }
    else
#endif
    {
#if defined(DEBUG) && !defined(CCTAG_NO_COUT)
        const auto optimizationStart = std::chrono::steady_clock::now();
#endif
        float neighbourSize = params._imagedCenterNeighbourSize;
        if(!std::isfinite(neighbourSize) || neighbourSize < 0.f)
            return false;
        const std::size_t gridSamples = params._imagedCenterNGridSample;
        // A smaller grid cannot shrink the search neighbourhood between passes.
        if(gridSamples < 5)
            return false;
        const float maxSemiAxis = std::max(outerEllipse.a(), outerEllipse.b());
        const float shrinkFactor = static_cast<float>((gridSamples - 1) / 2);

        bool searched = false;
        // Stop when the search width falls below 0.02 pixels.
        while(neighbourSize * maxSemiAxis > 0.02)
        {
            if(!imageCenterOptimizationGlob(
                 homography, cuts, optimalPoint, residual, neighbourSize, src, outerEllipse, params))
                return false;
            searched = true;
            visualDebug.drawPoint(optimalPoint, color_blue);
            neighbourSize /= shrinkFactor;
        }
        if(!searched)
        {
            // A zero or subpixel neighbourhood still needs a measured residual.
            computeHomographyFromEllipseAndImagedCenter(outerEllipse, optimalPoint, homography);
            bool readable;
            residual = costFunctionGlob(homography, cuts, src, readable);
            if(!readable)
                return false;
        }
#if defined(DEBUG) && !defined(CCTAG_NO_COUT)
        const auto spendTime =
          std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - optimizationStart)
            .count();
#endif
        DO_TALK(CCTAG_COUT_DEBUG("Optimization result: " << optimalPoint << ", duration: " << spendTime);)
    }
    visualDebug.drawPoint(optimalPoint, color_red);
    getSignals(cuts, homography, src);

    std::vector<const std::vector<float>*> readableSignals;
    readableSignals.reserve(cuts.size());
    for(const ImageCut& cut : cuts)
    {
        if(!cut.outOfBounds())
        {
            if(!readableSignals.empty() && cut.imgSignal().size() != readableSignals.front()->size())
                return false;
            readableSignals.push_back(&cut.imgSignal());
        }
    }

    if(readableSignals.size() < 2)
        return false;
    const std::size_t sampleCount = readableSignals.front()->size();

    std::vector<float> samples(readableSignals.size());
    const std::size_t medianSize = samples.size() / 2 + 1;
    std::vector<float> medianValues(medianSize);
    float minimumIntensity = 0.f;
    float maximumIntensity = 0.f;
    for(std::size_t sampleIndex = 0; sampleIndex < sampleCount; ++sampleIndex)
    {
        for(std::size_t cutIndex = 0; cutIndex < readableSignals.size(); ++cutIndex)
            samples[cutIndex] = (*readableSignals[cutIndex])[sampleIndex];

        std::partial_sort_copy(samples.begin(), samples.end(), medianValues.begin(), medianValues.end());
        const float medianIntensity = samples.size() % 2
                                        ? medianValues.back()
                                        : (medianValues[medianSize - 1] + medianValues[medianSize - 2]) / 2.0;
        if(sampleIndex == 0)
            minimumIntensity = maximumIntensity = medianIntensity;
        else
        {
            minimumIntensity = std::min(minimumIntensity, medianIntensity);
            maximumIntensity = std::max(maximumIntensity, medianIntensity);
        }
    }

    // Normalize by the contrast of the median profile across readable cuts.
    const float contrast = maximumIntensity - minimumIntensity;
    if(contrast <= 0.f)
        return false;
    residual = std::sqrt(residual) / contrast;
    return residual <= 2.7f;
}

bool imageCenterOptimizationGlob(Eigen::Matrix3f& homography,
                                 std::vector<ImageCut>& cuts,
                                 Point2d<Eigen::Vector3f>& center,
                                 float& minResidual,
                                 float neighbourSize,
                                 const cv::Mat& src,
                                 const numerical::geometry::Ellipse& outerEllipse,
                                 const Parameters& params)
{
    std::vector<Point2d<Eigen::Vector3f>> nearbyPoints;
    getNearbyPoints(outerEllipse, center, nearbyPoints, neighbourSize, params._imagedCenterNGridSample, GRID);

    minResidual = std::numeric_limits<float>::max();
    Point2d<Eigen::Vector3f> bestCenter;
    Eigen::Matrix3f bestHomography;
    bool foundSolution = false;
    for(const auto& point : nearbyPoints)
    {
        CCTagVisualDebug::instance().drawPoint(point, color_green);
        Eigen::Matrix3f candidateHomography;
        try
        {
            computeHomographyFromEllipseAndImagedCenter(outerEllipse, point, candidateHomography);
        }
        catch(...)
        {
            continue;
        }

        bool readable;
        const float candidateResidual = costFunctionGlob(candidateHomography, cuts, src, readable);
        if(readable && candidateResidual < minResidual)
        {
            foundSolution = true;
            minResidual = candidateResidual;
            bestCenter = point;
            bestHomography = candidateHomography;
        }
    }
    if(!foundSolution)
        return false;

    center = bestCenter;
    homography = bestHomography;
    return true;
}

void getNearbyPoints(const numerical::geometry::Ellipse& ellipse,
                     const Point2d<Eigen::Vector3f>& center,
                     std::vector<Point2d<Eigen::Vector3f>>& nearbyPoints,
                     float neighbourSize,
                     std::size_t gridNSample,
                     NeighborType neighborType)
{
    nearbyPoints.clear();
    if(neighborType != GRID)
        return;
    if(gridNSample < 2)
        throw std::invalid_argument("A nearby-point grid requires at least two samples per axis");
    if(gridNSample > nearbyPoints.max_size() / gridNSample)
        throw std::length_error("Nearby-point grid is too large");

    const Eigen::Matrix3f imageToConditioned = numerical::optimization::conditionerFromEllipse(ellipse);
    const Eigen::Matrix3f conditionedToImage = imageToConditioned.inverse();
    numerical::geometry::Ellipse conditionedEllipse(ellipse);
    // Conics transform with the inverse point transformation.
    viewGeometry::projectiveTransform(conditionedToImage, conditionedEllipse);
    neighbourSize *= std::max(conditionedEllipse.a(), conditionedEllipse.b());

    Point2d<Eigen::Vector3f> conditionedCenter = center;
    numerical::optimization::condition(conditionedCenter, imageToConditioned);
    const float halfWidth = neighbourSize / 2.f;
    const float stepSize = neighbourSize / (gridNSample - 1);
    nearbyPoints.reserve(gridNSample * gridNSample);
    for(std::size_t x = 0; x < gridNSample; ++x)
        for(std::size_t y = 0; y < gridNSample; ++y)
            nearbyPoints.emplace_back(conditionedCenter.x() - halfWidth + x * stepSize,
                                      conditionedCenter.y() - halfWidth + y * stepSize);
    numerical::optimization::condition(nearbyPoints, conditionedToImage);
}

// Sum squared sample differences, averaged over readable cut pairs.
float costFunctionGlob(const Eigen::Matrix3f& homography,
                       std::vector<ImageCut>& cuts,
                       const cv::Mat& image,
                       bool& readable)
{
    getSignals(cuts, homography, image);
    const std::vector<float>* firstReadableSignal = nullptr;
    std::size_t readableCount = 0;
    for(const auto& cut : cuts)
    {
        if(cut.outOfBounds())
            continue;
        if(!firstReadableSignal)
            firstReadableSignal = &cut.imgSignal();
        else if(cut.imgSignal().size() != firstReadableSignal->size())
        {
            readable = false;
            return std::numeric_limits<float>::max();
        }
        ++readableCount;
    }
    readable = readableCount >= 2;
    if(!readable)
        return std::numeric_limits<float>::max();

    const double count = static_cast<double>(readableCount);
    double totalPairSquares = 0.0;
    for(std::size_t sample = 0; sample < firstReadableSignal->size(); ++sample)
    {
        const double origin = (*firstReadableSignal)[sample];
        double sum = 0.0;
        double sumSquares = 0.0;
        for(const auto& cut : cuts)
        {
            if(cut.outOfBounds())
                continue;
            const double difference = static_cast<double>(cut.imgSignal()[sample]) - origin;
            sum += difference;
            sumSquares += difference * difference;
        }
        // sum((x_i - x_j)^2) over pairs equals count * sum(d^2) - sum(d)^2.
        // Shift d by the first signal and accumulate in double to retain small differences.
        totalPairSquares += count * sumSquares - sum * sum;
    }
    const double pairCount = count * (count - 1.0) / 2.0;
    return static_cast<float>(totalPairSquares / pairCount);
}

int identify_step_1(int /* tagIndex */,
                    const CCTag& tag,
                    std::vector<ImageCut>& selectedCuts,
                    const cv::Mat& image,
                    const Parameters& params)
{
    // Geometry is rescaled to match the original image.
    const auto& ellipse = tag.rescaledOuterEllipse();
    const auto& outerEllipsePoints = tag.rescaledOuterEllipsePoints();
#ifdef CCTAG_OPTIM
    auto startTime = std::chrono::steady_clock::now();
#endif
    std::vector<DirectedPoint2d<Eigen::Vector3f>> sampledOuterPoints;
    getSortedOuterPoints(ellipse, outerEllipsePoints, sampledOuterPoints, params._nSamplesOuterEllipse);
#ifdef CCTAG_OPTIM
    DO_TALK(CCTAG_COUT_OPTIM("Time in subsampling: " << std::chrono::duration_cast<std::chrono::milliseconds>(
                                                          std::chrono::steady_clock::now() - startTime)
                                                          .count()
                                                     << " ms");)
#endif
    if(sampledOuterPoints.size() < 5)
        return status::too_few_outer_points;

    for(const auto& point : sampledOuterPoints)
        CCTagVisualDebug::instance().drawPoint(Point2d<Eigen::Vector3f>(point.x(), point.y()), color_green);

    // Skip the unencoded white center of the marker.
    float beginSig = 0.f;
    if(params._nCrowns == 3)
        beginSig = 1 - (2 * params._nCrowns - 1) * 0.15f;
    else if(params._nCrowns == 4)
        beginSig = 0.26f;
    else
        CCTAG_COUT("Error : unknown number of crowns");

#ifdef CCTAG_OPTIM
    startTime = std::chrono::steady_clock::now();
#endif
    std::vector<ImageCut> cuts;
    collectCuts(cuts, image, ellipse.center(), sampledOuterPoints, params._sampleCutLength, beginSig);
#ifdef CCTAG_OPTIM
    DO_TALK(CCTAG_COUT_OPTIM("Time in collectCuts: " << std::chrono::duration_cast<std::chrono::milliseconds>(
                                                          std::chrono::steady_clock::now() - startTime)
                                                          .count()
                                                     << " ms");)
#endif
    if(cuts.empty())
        return status::no_collected_cuts;

    selectCutCheapUniform(selectedCuts,
                          params._numCutsInIdentStep,
                          ellipse,
                          cuts,
                          image,
                          tag.scale(),
                          params._numSamplesOuterEdgePointsRefinement);
    DO_TALK(CCTAG_COUT_OPTIM("Initial cut selection");)
    if(selectedCuts.empty())
    {
        CCTAG_COUT_DEBUG("Unable to select any cut.");
        return status::no_selected_cuts;
    }
    // Cuts are ready for step 2; the ID has not been read yet.
    return status::id_reliable;
}

int identify_step_2(int tagIndex,
                    CCTag& tag,
                    std::vector<ImageCut>& selectedCuts,
                    const RadiusRatioBank& radiusRatios,
                    const cv::Mat& src,
                    cctag::TagPipe* cudaPipe,
                    const Parameters& params)
{
    if(radiusRatios.empty())
        return status::id_not_reliable;

    const auto& outerEllipse = tag.rescaledOuterEllipse();
    float residual = std::numeric_limits<float>::max();
    const bool converged = refineConicFamilyGlob(tagIndex,
                                                 tag.homography(),
                                                 tag.centerImg(),
                                                 selectedCuts,
                                                 src,
                                                 cudaPipe,
                                                 outerEllipse,
                                                 params,
#ifdef CCTAG_WITH_CUDA
                                                 tag.getNearbyPointBuffer(),
#else
                                                 nullptr,
#endif
                                                 residual);
    tag.setQuality(1.f / residual);
    if(!converged)
    {
        DO_TALK(CCTAG_COUT_DEBUG("Optimization on imaged center failed to converge.");)
        return status::opti_has_diverged;
    }

    std::vector<std::list<float>> votesById(radiusRatios.size());
    if(!orazioDistanceRobust(votesById, radiusRatios, selectedCuts, params._minIdentProba))
        return status::id_not_reliable;

#ifdef CCTAG_VISUAL_DEBUG
    cv::Mat rectifiedSignals;
    createRectifiedCutImage(selectedCuts, rectifiedSignals);
    auto& visualDebug = CCTagVisualDebug::instance();
    visualDebug.initBackgroundImage(rectifiedSignals);
    visualDebug.newSession("rectifiedSignal" + std::to_string(visualDebug.getMarkerIndex()));
    visualDebug.incrementMarkerIndex();
    visualDebug.newSession("refineConicPts");
#endif

    std::size_t bestId = 0;
    std::size_t mostVotes = 0;
    for(std::size_t markerId = 0; markerId < votesById.size(); ++markerId)
    {
        if(votesById[markerId].size() > mostVotes)
        {
            bestId = markerId;
            mostVotes = votesById[markerId].size();
        }
    }
    if(mostVotes == 0)
        return status::id_not_reliable;

    float meanProbability = 0.f;
    for(float probability : votesById[bestId])
        meanProbability += probability;
    meanProbability /= mostVotes;

    tag.setId(static_cast<MarkerID>(bestId));
    tag.setIdSet({});
    tag.setRadiusRatios(radiusRatios[bestId]);
    try
    {
        const Eigen::Matrix3f imageToMarker = tag.homography().inverse();
        auto& ellipses = tag.ellipses();
        ellipses.reserve(ellipses.size() + tag.radiusRatios().size() + 1);
        for(float radiusRatio : tag.radiusRatios())
        {
            const numerical::geometry::Circle circle(1.f / radiusRatio);
            ellipses.emplace_back(imageToMarker.transpose() * circle.matrix() * imageToMarker);
        }
        ellipses.push_back(outerEllipse);
    }
    catch(...)
    {
        return status::degenerate;
    }
    return meanProbability > params._minIdentProba ? status::id_reliable : status::id_not_reliable;
}

} // namespace identification
} // namespace cctag
