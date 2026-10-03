@echo off
REM ============================================================================
REM  Build ninfer-serve.exe for sm_86 (RTX 30 series). SM count is detected at
REM  runtime, so one binary serves 3060 / 3070 / 3080 / 3090 and
REM  (binary-compatible) RTX 40 series.
REM
REM  ASCII-only on purpose: batch files are read in the OEM codepage.
REM  Edit the paths below. Build path must be pure ASCII.
REM   * VSLANG=1033 so MSVC emits English "Note: including file:" lines for ninja
REM   * /utf-8 for UTF-8 CUDA headers (otherwise C4819 floods on CP936 systems)
REM   * /EHsc, /DWIN32 /D_WINDOWS restored explicitly: passing -DCMAKE_CXX_FLAGS
REM     replaces CMake's default seed instead of appending to it
REM   * /MANIFEST:NO: a portable SDK layout ships no rc.exe
REM
REM  Dependencies (see swift15/README.md): CUDA 13.x with cuBLAS, MSVC 2022,
REM  CMake + Ninja, FFmpeg dev package, zlib, libcurl.
REM
REM  usage: build-sm86.bat [configure|build]   (no arg = configure then build)
REM ============================================================================
setlocal
set "TREE=%~dp0..\engine"
set "BUILD=%~dp0..\build"
set "CUDA=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3"
set "VCVARS=C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
set "DEPS=D:\build\deps"
set "FFMPEG_CFG=D:\build\ffmpeg-cfg"
set "CFLAGS=/DWIN32 /D_WINDOWS /utf-8"
set "CXXFLAGS=/DWIN32 /D_WINDOWS /EHsc /utf-8"
set "CUFLAGS=-D_WINDOWS -Xcompiler=/EHsc -Xcompiler=/utf-8"
set "NOMAN=/MANIFEST:NO"
set "VSLANG=1033"
call "%VCVARS%"
set "PATH=%CUDA%\bin;%CUDA%\bin\x64;%PATH%"
set "CUDA_PATH=%CUDA%"
where cl.exe    >nul 2>&1 || (echo [FATAL] cl.exe not on PATH   & exit /b 94)
where nvcc.exe  >nul 2>&1 || (echo [FATAL] nvcc.exe not on PATH & exit /b 95)
where cmake.exe >nul 2>&1 || (echo [FATAL] cmake not on PATH    & exit /b 96)
where ninja.exe >nul 2>&1 || (echo [FATAL] ninja not on PATH    & exit /b 97)
if /i "%~1"=="build" goto :build
:configure
cmake -B "%BUILD%" -S "%TREE%" -G Ninja ^
  -DCMAKE_BUILD_TYPE=Release ^
  -DCMAKE_CUDA_ARCHITECTURES=86 ^
  -DZLIB_ROOT="%DEPS%" ^
  -DFFMPEG_DIR="%FFMPEG_CFG%" ^
  -DCURL_INCLUDE_DIR="%DEPS%/include" -DCURL_LIBRARY="%DEPS%/lib/libcurl.lib" ^
  -DCMAKE_CUDA_COMPILER="%CUDA%/bin/nvcc.exe" ^
  -DCUDAToolkit_ROOT="%CUDA%" ^
  -DBUILD_TESTING=OFF ^
  -DNINFER_BUILD_BENCHMARKS=OFF ^
  -DNINFER_SLIM_3060=ON ^
  -DCMAKE_C_FLAGS="%CFLAGS%" ^
  -DCMAKE_CXX_FLAGS="%CXXFLAGS%" ^
  -DCMAKE_CUDA_FLAGS="%CUFLAGS%" ^
  -DCMAKE_EXE_LINKER_FLAGS="%NOMAN%" ^
  -DCMAKE_SHARED_LINKER_FLAGS="%NOMAN%" ^
  -DCMAKE_MODULE_LINKER_FLAGS="%NOMAN%"
if errorlevel 1 exit /b 1
if /i "%~1"=="configure" exit /b 0
:build
cmake --build "%BUILD%" -j 5 --target ninfer-serve
exit /b %ERRORLEVEL%
