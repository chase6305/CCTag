ARG CUDA_TAG=11.8.0
ARG OS_TAG=20.04
FROM alicevision/cctag-deps:cuda${CUDA_TAG}-ubuntu${OS_TAG}
LABEL maintainer="AliceVision Team alicevision@googlegroups.com"

# use CUDA_TAG to select the image version to use
# see https://hub.docker.com/r/nvidia/cuda/
#
# Build Dockerfile_deps first with the same CUDA_TAG and OS_TAG.
#
# then execute with nvidia docker (https://github.com/nvidia/nvidia-docker/wiki/Installation-(version-2.0))
# docker run -it --runtime=nvidia cctag


COPY . /opt/cctag
WORKDIR /opt/cctag/build
RUN cmake .. -DCCTAG_WITH_CUDA:BOOL=ON \
       -DCMAKE_BUILD_TYPE=Release \
       -DBUILD_SHARED_LIBS:BOOL=ON \
       -DCMAKE_PREFIX_PATH:PATH=/opt/ && make install -j$(nproc)
