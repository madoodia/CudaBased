@echo off
REM ============================================
REM  Build script for Step_02_VectorAdd
REM ============================================

REM --- Setup Visual Studio 18 Build Tools environment ---
call "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" x64

REM --- Paths ---
set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3
set SRC_DIR=src
set BUILD_DIR=build
set OUT_NAME=vector_add.exe

REM --- Create build dir ---
if not exist %BUILD_DIR% mkdir %BUILD_DIR%

REM --- Compile ---
echo Compiling Step 02: Vector Addition (main.cpp + kernel.cu)...
nvcc -g -G -arch=sm_86 -o %BUILD_DIR%\%OUT_NAME% %SRC_DIR%\main.cpp %SRC_DIR%\kernel.cu ^
     -I"%CUDA_PATH%\include" ^
     -L"%CUDA_PATH%\lib\x64" ^
     -lcudart ^
     -Xlinker /DEBUG

if %ERRORLEVEL% neq 0 (
    echo Compilation failed.
    exit /b %ERRORLEVEL%
)

REM --- Move debug files to the build directory ---
if exist vc*.pdb move /y vc*.pdb %BUILD_DIR%\ >nul

echo Build complete: %BUILD_DIR%\%OUT_NAME%
