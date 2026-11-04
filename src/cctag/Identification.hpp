/*
 * Copyright 2016, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#pragma once

#include <cctag/utils/VisualDebug.hpp>
#include <cctag/EllipseGrowing.hpp>
#include <cctag/ImageCut.hpp>
#include <cctag/geometry/Ellipse.hpp>
#include <cctag/geometry/Distance.hpp>
#include <cctag/Statistic.hpp>

#include <opencv2/opencv.hpp>

#include <cctag/utils/Math.hpp>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace cctag {
class TagPipe;
class NearbyPoint;
};

namespace cctag {
namespace identification {

enum NeighborType {
  GRID,
  CIRCULAR,
  COV
};

/// Collect and select radial cuts from the original-scale grayscale image.
/// Returns status::id_reliable when the cuts are ready for step 2; no ID or
/// optimized geometry is assigned in this step. Other statuses describe failure.
int identify_step_1(
  int tagIndex, const CCTag& tag, std::vector<ImageCut>& selectedCuts, const cv::Mat& image, const Parameters& params);

/// Refine the center and marker-to-image homography, then identify the rectified
/// profiles against radiusRatios. Returns a marker status from CCTag.hpp.
int identify_step_2(int tagIndex,
                    CCTag& tag,
                    std::vector<ImageCut>& selectedCuts,
                    const std::vector<std::vector<float>>& radiusRatios,
                    const cv::Mat& image,
                    TagPipe* cudaPipe,
                    const Parameters& params);

using RadiusRatioBank = std::vector<std::vector<float>>;
using CutSelectionVec =  std::vector< std::pair< cctag::Point2d<Eigen::Vector3f>, cctag::ImageCut>>;

/**
 * @brief Apply a planar homography to a 2D point.
 * 
 * @param[out] xRes x coordinate of the transformed point
 * @param[out] yRes y coordinate of the transformed point
 * @param[in] mHomography the transformation represented by a 3x3 matrix
 * @param[in] x coordinate of the point to be transformed
 * @param[in] y coordinate of the point to be transformed
 */
inline void applyHomography(
        float & xRes,
        float & yRes,
        const Eigen::Matrix3f & mHomography,
        const float x,
        const float y)
{
  float u = mHomography(0,0)*x + mHomography(0,1)*y + mHomography(0,2);
  float v = mHomography(1,0)*x + mHomography(1,1)*y + mHomography(1,2);
  float w = mHomography(2,0)*x + mHomography(2,1)*y + mHomography(2,2);
  xRes = u/w;
  yRes = v/w;
}

/// Append each readable cut's best matching probability to the corresponding ID.
/// Cuts need more than 30 samples, finite statistics and both light and dark
/// samples. Flat or invalid profiles contribute no votes. Equal probabilities
/// select the last ID in the bank. The caller applies minIdentProba; this argument
/// is retained for compatibility. Returns false for empty cuts or an empty bank.
bool orazioDistanceRobust(std::vector<std::list<float>>& votesById,
                          const RadiusRatioBank& radiusRatios,
                          const std::vector<ImageCut>& cuts,
                          float minIdentProba);

/**
 * @brief Extract a rectified 1D signal along an image cut based on an homography.
 *
 * @param[out] cut image cut holding the rectified image signal
 * @param[in] src source grayscale image (uchar)
 * @param[in] mHomography cctag->image homography
 * @param[in] mInvHomography image->cctag homography
 * @note Fewer than two samples marks the cut as out of bounds.
 */
void extractSignalUsingHomography(
        cctag::ImageCut & cut,
        const cv::Mat & src,
        const Eigen::Matrix3f & mHomography,
        const Eigen::Matrix3f & mInvHomography);

/* deprecated */
void extractSignalUsingHomographyDeprec(
        cctag::ImageCut & rectifiedCut,
        const cv::Mat & src,
        Eigen::Matrix3f & mHomography,
        std::size_t nSamples = 100,
        float begin = 0.f,
        float end = 1.f );

/**
 * @brief Extract a regularly sampled 1D signal along an image cut.
 * 
 * @param[out] cut image cut that will hold the 1D image signal regularly 
 *             collected from cut.beginSig() to cut.endSig()
 * @param[in] src source gray scale image (uchar)
 * @note Fewer than two samples marks the cut as out of bounds.
 */
void cutInterpolated(
        cctag::ImageCut & cut,
        const cv::Mat & src);

/// Return the first convolution maximum and its sample index.
/// Throws std::invalid_argument when the image signal is empty.
std::pair<float,float> convImageCut(const std::vector<float> & kernel, ImageCut & cut);

/// Smooth with the legacy fixed kernel; sigma is retained for compatibility.
void blurImageCut(float sigma, cctag::ImageCut & cut);

bool outerEdgeRefinement(ImageCut & cut, const cv::Mat & src, float scale, size_t numSamplesOuterEdgePointsRefinement);

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
void collectCuts(std::vector<cctag::ImageCut>& cuts,
                 const cv::Mat& src,
                 const cctag::Point2d<Eigen::Vector3f>& center,
                 const std::vector<cctag::DirectedPoint2d<Eigen::Vector3f>>& outerPoints,
                 std::size_t nSamplesInCut,
                 float beginSig);

/// Sample nonnegative image coordinates with a complete 2x2 neighbourhood.
/// The signal convention uses half-scale grayscale intensity.
inline float getPixelBilinear(const cv::Mat& src, float x, float y)
{
    const int pixelX = static_cast<int>(x);
    const int pixelY = static_cast<int>(y);
    const uchar* top = src.ptr<uchar>(pixelY) + pixelX;
    const uchar* bottom = top + src.step;
    const float fractionX = x - pixelX;
    const float fractionY = y - pixelY;
    const float inverseX = 1.f - fractionX;
    const float inverseY = 1.f - fractionY;
    const float topLeftWeight = inverseX * inverseY;
    const float topRightWeight = fractionX * inverseY;
    const float bottomLeftWeight = inverseX * fractionY;
    const float bottomRightWeight = fractionX * fractionY;
    return (top[0] * topLeftWeight + top[1] * topRightWeight + bottom[0] * bottomLeftWeight +
            bottom[1] * bottomRightWeight) /
           2;
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
        const cv::Mat & src);

/**
 * @brief Compute the optimal homography/imaged center based on the
 * signal in the image and  the outer ellipse, supposed to be image the unit circle.
 *
 * @param[in] tagIndex a sequence number for this tag
 * @param[out] mHomography cctag->image homography to optimize
 * @param[in,out] optimalPoint initial imaged center, replaced by the optimized center
 * @param[in,out] vCuts selected cuts, holding the rectified signals on success
 * @param[out] residual from the optimization (normalized w.r.t. binary pattern)
 * @param[in] src source image
 * @param[in] outerEllipse outer ellipse
 * @param[in] params parameters of the cctag algorithm
 * @return true if the optimization has found a solution, false otherwise.
 * The CPU path evaluates the initial center when the search width is already
 * below 0.02 pixels, including a zero neighbourhood size.
 */
bool refineConicFamilyGlob(
        int tagIndex,
        Eigen::Matrix3f & mHomography,
        Point2d<Eigen::Vector3f> & optimalPoint,
        std::vector< cctag::ImageCut > & vCuts, 
        const cv::Mat & src,
        cctag::TagPipe* cudaPipe,
        const cctag::numerical::geometry::Ellipse & outerEllipse,
        const cctag::Parameters & params,
        cctag::NearbyPoint* cctag_pointer_buffer,
        float & residual);

/**
 * @brief Convex optimization of the imaged center within a point's neighbourhood.
 *
 * @param[out] mHomography optimal homography from the cctag plane to the pixel plane.
 * @param[out] vCuts scratch signals from the last evaluated candidate; call
 *                  getSignals with the returned homography to sample the optimum
 * @param[out] center optimal imaged center
 * @param[out] minRes residual after optimization
 * @param[in] neighbourSize size of the neighbourhood to consider relatively to the outer ellipse dimensions
 * @param[in] src source gray (uchar) image
 * @param[in] outerEllipse outer ellipse
 * @param[in] params Parameters read from config file
 */
bool imageCenterOptimizationGlob(
        Eigen::Matrix3f & mHomography,
        std::vector< cctag::ImageCut > & vCuts,
        cctag::Point2d<Eigen::Vector3f> & center,
        float & minRes,
        float neighbourSize,
        const cv::Mat & src, 
        const cctag::numerical::geometry::Ellipse & outerEllipse,
        const cctag::Parameters & params );

/// Replace nearbyPoints with a square GRID centered at center. neighbourSize
/// sets its width relative to the largest ellipse semi-axis; gridNSample is the
/// number of points per axis and must be at least two. Other patterns are not
/// implemented and produce no points. Throws for invalid or overflowing sizes.
void getNearbyPoints(
          const cctag::numerical::geometry::Ellipse & ellipse,
          const cctag::Point2d<Eigen::Vector3f> & center,
          std::vector<cctag::Point2d<Eigen::Vector3f> > & nearbyPoints,
          float neighbourSize,
          std::size_t gridNSample,
          NeighborType neighborType);

/**
 * @brief Compute an homography (up to a 2D rotation) based on its imaged origin [0,0,1]'
 * and its imaged unit circle (represented as an ellipse, assuming only quasi-affine transformation
 * 
 * @param[in] ellipse ellipse image of the unit circle
 * @param[in] center imaged center, projection of the origin
 * @param[out] mHomography computed homography
 */
void computeHomographyFromEllipseAndImagedCenter(
        const cctag::numerical::geometry::Ellipse & ellipse,
        const cctag::Point2d<Eigen::Vector3f> & center,
        Eigen::Matrix3f & mHomography);

/// Resample the cuts and return summed squared sample differences, averaged
/// over readable cut pairs. Readable cuts need equal sample counts.
/// Set readable to false and return max<float> for mismatched sample counts
/// or fewer than two readable cuts.
float costFunctionGlob(const Eigen::Matrix3f& homography,
                       std::vector<ImageCut>& cuts,
                       const cv::Mat& image,
                       bool& readable);

/// Compute the median without modifying the input; average the central pair
/// for even sample counts. Throws std::invalid_argument for an empty input.
template<typename VecT>
typename VecT::value_type computeMedian(const VecT& values)
{
    using Value = typename VecT::value_type;
    if(values.empty())
        throw std::invalid_argument("computeMedian requires at least one sample");

    const std::size_t medianSize = values.size() / 2 + 1;
    std::vector<Value> lowerHalf(medianSize);
    std::partial_sort_copy(values.begin(), values.end(), lowerHalf.begin(), lowerHalf.end());
    if(values.size() % 2)
        return lowerHalf.back();
    return (lowerHalf[medianSize - 1] + lowerHalf[medianSize - 2]) / 2.0;
}

inline float dis( const float sig, const float val, const float mub, const float muw, const float varSubS )
{
  if ( val == -1 )
  {
          return cctag::numerical::square( std::max( sig - mub, 0.f ) ) / ( 2.0f * varSubS );
  }
  else
  {
          return cctag::numerical::square( std::min( sig - muw, 0.f ) ) / ( 2.0f * varSubS );
  }
}

/**
 * @deprecated
 */
bool refineConicFamily(
        CCTag & cctag,
        std::vector< cctag::ImageCut > & fsig,
        std::size_t lengthSig,
        const cv::Mat & src,
        const cctag::numerical::geometry::Ellipse & ellipse,
        const std::vector< cctag::Point2d<Eigen::Vector3f> > & pr,
        bool useLmDif );

/**
 * Get a point transformed by an homography
 * @param xi
 * @param yi
 * @param mH
 * @deprecated
 */
inline cctag::Point2d<Eigen::Vector3f> getHPoint(
        const float xi,
        const float yi,
        const Eigen::Matrix3f& mH )
{
  using namespace cctag::numerical;
  Eigen::Vector3f vh;
  vh( 0 ) = xi; vh( 1 ) = yi; vh( 2 ) = 1.f;

  return cctag::Point2d<Eigen::Vector3f>(mH * vh);
}

/**
 * @deprecated
 */
bool orazioDistance(
        IdSet& idSet,
        const RadiusRatioBank & rrBank,
        const std::vector<cctag::ImageCut> & cuts,
        std::size_t startOffset,
        float minIdentProba,
        std::size_t sizeIds);

} // namespace identification
} // namespace cctag

