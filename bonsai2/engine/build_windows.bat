@echo off
echo ========================================================
echo NInfer Build Script (Windows MSVC)
echo ========================================================

if not exist ffmpeg (
    echo [1/4] Downloading FFmpeg Windows dev binaries...
    powershell -Command "Invoke-WebRequest -Uri 'https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl-shared.zip' -OutFile 'ffmpeg.zip'"
    
    echo [2/4] Extracting FFmpeg...
    powershell -Command "Expand-Archive -Path 'ffmpeg.zip' -DestinationPath 'ffmpeg_temp' -Force"
    
    echo Moving files into place...
    move ffmpeg_temp\ffmpeg-master-latest-win64-gpl-shared ffmpeg >nul
    
    echo Cleaning up...
    rmdir /S /Q ffmpeg_temp
    del ffmpeg.zip
) else (
    echo [1/4] FFmpeg directory already exists. Skipping download.
)

echo.
echo [3/4] Setting up MSVC environment...
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" 2>nul || call "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat" 2>nul || call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" 2>nul || call "C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvars64.bat" 2>nul || call "C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat" 2>nul || call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat" 2>nul || echo Please open x64 Native Tools Command Prompt manually!

echo.
echo [4/4] Compiling NInfer...
cmake -B build -S . -G Ninja -DCMAKE_CUDA_ARCHITECTURES="120a" -DNINFER_ENABLE_AVX2=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release

echo.
echo Copying FFmpeg DLLs to the build folder so the executable can find them...
copy ffmpeg\bin\avcodec-*.dll build\apps\ >nul 2>nul
copy ffmpeg\bin\avformat-*.dll build\apps\ >nul 2>nul
copy ffmpeg\bin\avutil-*.dll build\apps\ >nul 2>nul
copy ffmpeg\bin\swscale-*.dll build\apps\ >nul 2>nul
copy ffmpeg\bin\swresample-*.dll build\apps\ >nul 2>nul

echo.
echo ========================================================
echo Build Complete!
echo Executable is located at: build\apps\ninfer-serve.exe
echo ========================================================
pause
