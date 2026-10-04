@echo off
REM Build the Bonsai2 12GB engine for sm_86 on Windows.
REM ASCII-only batch source. Set the toolchain paths below or via environment.
REM Usage: build-sm86.bat [configure^|build]
setlocal
set "TREE=%~dp0..\engine"
set "BUILD=%~dp0..\build"
if not defined NINFER_CUDA_PATH set "NINFER_CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3"
if not defined NINFER_VCVARS64 set "NINFER_VCVARS64=C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
if not defined NINFER_DISABLE_MEDIA set "NINFER_DISABLE_MEDIA=OFF"
if not defined NINFER_BUILD_TESTING set "NINFER_BUILD_TESTING=OFF"
if not defined NINFER_BUILD_BENCHMARKS set "NINFER_BUILD_BENCHMARKS=OFF"
if not defined NINFER_BUILD_JOBS set "NINFER_BUILD_JOBS=5"
if not exist "%NINFER_VCVARS64%" (
  echo [FATAL] Set NINFER_VCVARS64 to your vcvars64.bat.
  exit /b 93
)
set "VSLANG=1033"
call "%NINFER_VCVARS64%"
if errorlevel 1 exit /b 93
if defined NINFER_TOOLS_PATH set "PATH=%NINFER_TOOLS_PATH%;%PATH%"
set "PATH=%NINFER_CUDA_PATH%\bin;%NINFER_CUDA_PATH%\bin\x64;%PATH%"
set "CUDA_PATH=%NINFER_CUDA_PATH%"
where cl.exe    >nul 2>&1 || (echo [FATAL] cl.exe not on PATH   & exit /b 94)
where nvcc.exe  >nul 2>&1 || (echo [FATAL] nvcc.exe not on PATH & exit /b 95)
where cmake.exe >nul 2>&1 || (echo [FATAL] cmake not on PATH    & exit /b 96)
where ninja.exe >nul 2>&1 || (echo [FATAL] ninja not on PATH    & exit /b 97)
if /i "%~1"=="build" goto :build
if /i "%NINFER_DISABLE_MEDIA%"=="OFF" (
  if not exist "%TREE%\ffmpeg\include\libavcodec\avcodec.h" (
    echo [FATAL] Put the FFmpeg development include and lib directories in engine\ffmpeg.
    echo         Or set NINFER_DISABLE_MEDIA=ON for a text-only build.
    exit /b 98
  )
  if not exist "%TREE%\ffmpeg\lib\avcodec.lib" (
    echo [FATAL] FFmpeg import libraries are missing from engine\ffmpeg\lib.
    exit /b 98
  )
)
cmake -B "%BUILD%" -S "%TREE%" -G Ninja ^
  -DCMAKE_BUILD_TYPE=Release ^
  -DCMAKE_CUDA_ARCHITECTURES=86 ^
  -DCMAKE_CUDA_COMPILER="%NINFER_CUDA_PATH%/bin/nvcc.exe" ^
  -DCUDAToolkit_ROOT="%NINFER_CUDA_PATH%" ^
  -DNINFER_BUILD_APPS=ON ^
  -DNINFER_DISABLE_MEDIA=%NINFER_DISABLE_MEDIA% ^
  -DNINFER_RK4_SM86=OFF ^
  -DBUILD_TESTING=%NINFER_BUILD_TESTING% ^
  -DNINFER_BUILD_BENCHMARKS=%NINFER_BUILD_BENCHMARKS% ^
  -DCMAKE_C_FLAGS="/DWIN32 /D_WINDOWS /utf-8" ^
  -DCMAKE_CXX_FLAGS="/DWIN32 /D_WINDOWS /EHsc /utf-8" ^
  -DCMAKE_CUDA_FLAGS="-D_WINDOWS -Xcompiler=/EHsc -Xcompiler=/utf-8" ^
  -DCMAKE_EXE_LINKER_FLAGS="/MANIFEST:NO" ^
  -DCMAKE_SHARED_LINKER_FLAGS="/MANIFEST:NO" ^
  -DCMAKE_MODULE_LINKER_FLAGS="/MANIFEST:NO"
if errorlevel 1 exit /b 1
if /i "%~1"=="configure" exit /b 0
:build
cmake --build "%BUILD%" -j %NINFER_BUILD_JOBS% --target ninfer-serve ninfer-perplexity
exit /b %ERRORLEVEL%
