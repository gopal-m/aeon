/*
 Copyright 2016 Nervana Systems Inc.
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
*/

#pragma once

#include <tuple>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

namespace nervana {
    namespace image {
        // These functions may be common across different transformers
        void resize(const cv::Mat&, cv::Mat&, const cv::Size2i&, bool interpolate=true);
        void rotate(const cv::Mat& input, cv::Mat& output, int angle, bool interpolate=true, const cv::Scalar& border=cv::Scalar());
        void convert_mix_channels(std::vector<cv::Mat>& source, std::vector<cv::Mat>& target, std::vector<int>& from_to);

        std::tuple<float,cv::Size> calculate_scale_shape(cv::Size size, int min_size, int max_size);

        cv::Size2f cropbox_max_proportional(const cv::Size2f& in_size, const cv::Size2f& out_size);
        cv::Size2f cropbox_linear_scale(const cv::Size2f& in_size, float scale);
        cv::Size2f cropbox_area_scale(const cv::Size2f& in_size, const cv::Size2f& cropbox_size, float scale);
        cv::Point2f cropbox_shift(const cv::Size2f&, const cv::Size2f&, float, float);

        class photometric {
        public:
            photometric();
            void lighting(cv::Mat& inout, std::vector<float>, float color_noise_std);
            void cbsjitter(cv::Mat& inout, const std::vector<float>&);

            // These are the eigenvectors of the pixelwise covariance matrix
            const float _CPCA[3][3];
            const cv::Mat CPCA;

            // These are the square roots of the eigenvalues of the pixelwise covariance matrix
            const cv::Mat CSTD;

            // This is the set of coefficients for converting BGR to grayscale
            const cv::Mat GSCL;
        };
    }
}