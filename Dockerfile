# Use a specific version of Debian for stability
FROM debian:buster

# Avoid interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Set up working directory
WORKDIR /build

# Install essential packages
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    pkg-config \
    yasm \
    wget \
    python3 \
    python3-pip \
    libx264-dev \
    libglew-dev \
    libglfw3-dev \
    libgl1-mesa-dev \
    xvfb \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Clone required repositories with specific versions
RUN git clone https://git.ffmpeg.org/ffmpeg.git && \
    cd ffmpeg && \
    git checkout n4.2.2 && \
    cd .. && \
    git clone https://github.com/transitive-bullshit/ffmpeg-gl-transition.git

# Copy the gltransition filter to FFmpeg
RUN cp /build/ffmpeg-gl-transition/vf_gltransition.c /build/ffmpeg/libavfilter/

# Manually add entries to Makefile and allfilters.c
RUN cd /build/ffmpeg && \
    sed -i '/OBJS-$(CONFIG_ZOOMPAN_FILTER)/a OBJS-$(CONFIG_GLTRANSITION_FILTER) += vf_gltransition.o' libavfilter/Makefile && \
    sed -i '/extern AVFilter ff_vf_zoompan;/a extern AVFilter ff_vf_gltransition;' libavfilter/allfilters.c

# Configure and build FFmpeg with gltransition support
RUN cd /build/ffmpeg && \
    ./configure \
        --prefix=/usr/local \
        --enable-gpl \
        --enable-libx264 \
        --enable-opengl \
        --enable-filter=gltransition \
        --extra-libs='-lGLEW -lglfw -ldl' && \
    make -j$(nproc) && \
    make install && \
    ldconfig

# Create a transition directory
RUN mkdir -p /app/transitions && \
    echo 'vec4 transition(vec2 uv) {\n\
  return mix(\n\
    getFromColor(uv),\n\
    getToColor(uv),\n\
    progress\n\
  );\n\
}' > /app/transitions/fade.glsl

# Create a simple wrapper script for Xvfb
RUN echo '#!/bin/bash\n\
# Start Xvfb\n\
Xvfb :1 -screen 0 1280x1024x16 &\n\
XVFB_PID=$!\n\
sleep 2\n\
export DISPLAY=:1\n\
\n\
# Show FFmpeg filters to check if gltransition is available\n\
echo "Checking for gltransition filter:"\n\
ffmpeg -filters | grep gltransition\n\
\n\
# Execute the command\n\
echo "Running FFmpeg command:"\n\
echo ffmpeg -i "$1" -i "$2" -filter_complex "gltransition=duration=$4:offset=0:source=/app/transitions/$3" -y "$5"\n\
ffmpeg -i "$1" -i "$2" -filter_complex "gltransition=duration=$4:offset=0:source=/app/transitions/$3" -y "$5"\n\
\n\
# Clean up\n\
kill $XVFB_PID\n\
' > /app/run-transition.sh && chmod +x /app/run-transition.sh

# Set working directory
WORKDIR /app

# Set the entry point to the wrapper script
ENTRYPOINT ["/app/run-transition.sh"]