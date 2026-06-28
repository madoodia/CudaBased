@echo off
REM ============================================
REM  Profiling script for Step_02_VectorAdd
REM ============================================

REM --- Setup Visual Studio 18 Build Tools environment ---
call "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" x64

REM --- Paths ---
set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3
set SRC_DIR=src
set BUILD_DIR=build
set OUT_NAME=kernel.ptx

nvcc -ptx src/kernel.cu -o build/kernel.ptx -I"%CUDA_PATH%\include"

@REM or using NSight Compute (GUI, or Commandline)
@REM ncu --section SourceConfig --section SpeedOfLight build\vector_add.exe


