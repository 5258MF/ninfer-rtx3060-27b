@echo off
rem ============================================================
rem NInfer v1.0.8 (upstream b88c0f6f + fork ports 2026-09-09) -
rem RTX 4090 (sm_89), vision.
rem Autocontenido: solo necesita este arbol + MSVC BuildTools +
rem CUDA 13.3 + Ninja (los tres ya instalados; rutas en el script).
rem
rem Uso:
rem   build_v1.0.8.bat              -> build en _build_4090new (relativo al arbol)
rem   build_v1.0.8.bat <build_dir>  -> build en el directorio indicado
rem Sin "pause": apto para correr en background y redirigir log.
rem ============================================================
setlocal
set BUILD_DIR=%~1
if "%BUILD_DIR%"=="" set BUILD_DIR=%~dp0\_build_4090new
cd /d "%~dp0"

rem --- MSVC vcvars64 (BuildTools primero; fallbacks por si acaso) ---
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
if errorlevel 1 call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
if errorlevel 1 call "C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
if errorlevel 1 call "C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
echo [cfg] vcvars exit=%errorlevel%
if errorlevel 1 exit /b 90

rem --- Ninja (el que trae CMake de BuildTools) ---
set PATH=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja;%PATH%

echo [build] dir=%BUILD_DIR% arch=89
cmake -B "%BUILD_DIR%" -S . -G Ninja -DCMAKE_CUDA_ARCHITECTURES=89 -DNINFER_ENABLE_AVX2=ON -DNINFER_BUILD_MEDIA_ACQUIRE=ON -DCMAKE_BUILD_TYPE=Release
if errorlevel 1 exit /b 1
cmake --build "%BUILD_DIR%" --config Release -j 32
if errorlevel 1 exit /b 1

rem --- Artefactos a la raiz del build dir (layout de runtime) ---
copy /Y "%BUILD_DIR%\apps\ninfer-serve.exe" "%BUILD_DIR%\" >nul
copy /Y "%BUILD_DIR%\apps\ninfer.exe" "%BUILD_DIR%\" >nul
copy /Y "%BUILD_DIR%\apps\ninfer-perplexity.exe" "%BUILD_DIR%\" >nul
copy /Y "%~dp0ffmpeg\bin\avcodec-*.dll" "%BUILD_DIR%\" >nul
copy /Y "%~dp0ffmpeg\bin\avformat-*.dll" "%BUILD_DIR%\" >nul
copy /Y "%~dp0ffmpeg\bin\avutil-*.dll" "%BUILD_DIR%\" >nul
copy /Y "%~dp0ffmpeg\bin\swscale-*.dll" "%BUILD_DIR%\" >nul
copy /Y "%~dp0ffmpeg\bin\swresample-*.dll" "%BUILD_DIR%\" >nul
echo BUILD_4090_OK %BUILD_DIR%
exit /b 0
