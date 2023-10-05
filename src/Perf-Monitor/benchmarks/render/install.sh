#!/bin/bash
set -e -x

# manually compile: sdl2, nasm, x264, fdk-ac, libvpx, lame-3, opus

sudo apt-get update
sudo apt-get -y install python3-pip
sudo pip3 install --upgrade cython

# ffmpeg requirements
sudo apt-get update -qq && sudo apt-get -y install \
  autoconf \
  automake \
  build-essential \
  cmake \
  git-core \
  libass-dev \
  libfreetype6-dev \
  libgnutls28-dev \
  libtool \
  libva-dev \
  libvdpau-dev \
  libvorbis-dev \
  libxcb1-dev \
  libxcb-shm0-dev \
  libxcb-xfixes0-dev \
  meson \
  ninja-build \
  pkg-config \
  texinfo \
  wget \
  yasm \
  zlib1g-dev

sudo apt-get -y install libass-dev libfreetype6-dev libtheora-dev libvorbis-dev;

# clear and create source folder
sudo rm -rf ~/ffmpeg_sources ~/bin
mkdir -p ~/ffmpeg_sources ~/bin
# export lib path
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/ffmpeg_build/lib;

# sdl2
cd ~/ffmpeg_sources;
# backup resource https://ftp.osuosl.org/pub/blfs/conglomeration/SDL/SDL2-2.0.12.tar.gz
wget https://www.libsdl.org/release/SDL2-2.0.12.tar.gz;
tar xzf SDL2-2.0.12.tar.gz;
cd SDL2-2.0.12;
./configure --prefix="$HOME/ffmpeg_build" --bindir="$HOME/ffmpeg_build/bin";
make -j;
make install;
make distclean;

# nasm
cd ~/ffmpeg_sources && \
wget https://www.nasm.us/pub/nasm/releasebuilds/2.15.05/nasm-2.15.05.tar.bz2 && \
tar xjvf nasm-2.15.05.tar.bz2 && \
cd nasm-2.15.05 && \
./autogen.sh && \
PATH="$HOME/bin:$PATH" ./configure --prefix="$HOME/ffmpeg_build" --bindir="$HOME/ffmpeg_build/bin" && \
make -j && \
make install

# x264
cd ~/ffmpeg_sources && \
git -C x264 pull 2> /dev/null || git clone --depth 1 https://code.videolan.org/videolan/x264.git && \
cd x264 && \
PATH="$HOME/bin:$PATH" PKG_CONFIG_PATH="$HOME/ffmpeg_build/lib/pkgconfig" ./configure --prefix="$HOME/ffmpeg_build" --bindir="$HOME/ffmpeg_build/bin" --enable-shared --extra-cflags="-fPIC" && \
PATH="$HOME/bin:$PATH" make -j && \
make install


# sudo apt-get install libnuma-dev && \
# cd ~/ffmpeg_sources && \
# git -C x265_git pull 2> /dev/null || git clone https://bitbucket.org/multicoreware/x265_git && \
# cd x265_git/build/linux && \
# PATH="$HOME/bin:$PATH" cmake -G "Unix Makefiles" -DCMAKE_INSTALL_PREFIX="$HOME/ffmpeg_build" -DENABLE_SHARED:bool=on ../../source && \
# PATH="$HOME/bin:$PATH" make -j && \
# make install

# libvpx
cd ~/ffmpeg_sources && \
git -C libvpx pull 2> /dev/null || git clone --depth 1 https://chromium.googlesource.com/webm/libvpx.git && \
cd libvpx && \
PATH="$HOME/bin:$PATH" ./configure --prefix="$HOME/ffmpeg_build" --disable-examples  --as=yasm --enable-shared --disable-unit-tests && \
PATH="$HOME/bin:$PATH" make -j && \
make install

# fdk-acc
cd ~/ffmpeg_sources && \
git -C fdk-aac pull 2> /dev/null || git clone --depth 1 https://github.com/mstorsjo/fdk-aac && \
cd fdk-aac && \
autoreconf -fiv && \
./configure --prefix="$HOME/ffmpeg_build" --disable-shared && \
make -j && \
make install

# lame-3
cd ~/ffmpeg_sources && \
wget -O lame-3.100.tar.gz https://downloads.sourceforge.net/project/lame/lame/3.100/lame-3.100.tar.gz && \
tar xzvf lame-3.100.tar.gz && \
cd lame-3.100 && \
PATH="$HOME/bin:$PATH" ./configure --prefix="$HOME/ffmpeg_build" --enable-nasm --enable-shared && \
PATH="$HOME/bin:$PATH" make -j && \
make install

# opus
cd ~/ffmpeg_sources && \
git -C opus pull 2> /dev/null || git clone --depth 1 https://github.com/xiph/opus.git && \
cd opus && \
./autogen.sh && \
./configure --prefix="$HOME/ffmpeg_build" --enable-shared && \
make -j && \
make install

# ffmpeg version 4.3
cd ~/ffmpeg_sources;
wget http://ffmpeg.org/releases/ffmpeg-4.3.tar.bz2;
tar xjf ffmpeg-4.3.tar.bz2;
cd ffmpeg-4.3;
PATH="$HOME/ffmpeg_build/bin:$PATH" PKG_CONFIG_PATH="$HOME/ffmpeg_build/lib/pkgconfig:/usr/lib/pkgconfig/" ./configure --prefix="$HOME/ffmpeg_build" --extra-cflags="-I$HOME/ffmpeg_build/include -fPIC" --extra-ldflags="-L$HOME/ffmpeg_build/lib" --bindir="$HOME/ffmpeg_build/bin" --enable-gpl --enable-libass --enable-libfreetype --enable-libmp3lame --enable-libtheora --enable-libvorbis --enable-libx264 --enable-shared
PATH="$HOME/ffmpeg_build/bin:$PATH" make -j;
make install;
make distclean;

# export lib library and pkg config path
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/ffmpeg_build/lib
export PKG_CONFIG_PATH=$HOME/ffmpeg_build/lib/pkgconfig:$PKG_CONFIG_PATH

# compile from source git
sudo apt-get -y install libcanberra-gtk-module
cd ~/ffmpeg_sources;
rm -rf ffpyplayer
git clone https://github.com/Vampire-Vx/ffpyplayer.git
cd ffpyplayer
pip3 install -e .

# add export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/ffmpeg_build/lib to ~/.bashrc and source ~/.bashrc if needed