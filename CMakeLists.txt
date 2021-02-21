############################################################################
# <summary> CMakeLists.txt for OpenCV and CUDA. </summary>
# <date>    2012-02-07          </date>
# <author>  Quan Tran Minh. edit by Johannes Kast, Michael Sarahan </author>
# <email>   quantm@unist.ac.kr  kast.jo@googlemail.com msarahan@gmail.com</email>
############################################################################

cmake_minimum_required(VERSION 2.6 FATAL_ERROR)
project(fractal)

find_package(CUDA REQUIRED)
find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)
include_directories( ${OPENGL_INCLUDE_DIRS}  ${GLUT_INCLUDE_DIRS} )
set (EXECUTABLE_OUTPUT_PATH "${CMAKE_SOURCE_DIR}/bin/")

if(CUDA_FOUND)
    # compared to class settings, we let NVidia's FindCUDA CMake detect
    # whether to build x64.  We tell it to support most devices, though,
    # to make sure more people can easily run class code without knowing
    # about this compiler argument
    set(CUDA_NVCC_FLAGS "
  -ccbin /usr/bin/g++;
  -gencode;arch=compute_35,code=sm_35;
  -gencode;arch=compute_37,code=sm_37;
  -gencode;arch=compute_50,code=sm_50;
  -gencode;arch=compute_52,code=sm_52;
  -gencode;arch=compute_60,code=sm_60;
  -gencode;arch=compute_61,code=sm_61;
  -gencode;arch=compute_70,code=sm_70;
  --expt-relaxed-constexpr;")

    # add -Wextra compiler flag for gcc compilations
    if (UNIX)
        set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "-Xcompiler -Wextra")
        #        set(CMAKE_CXX_FLAGS "-stdlib=libstdc++")
    endif (UNIX)

    # add debugging to CUDA NVCC flags.  For NVidia's NSight tools.
    set(CUDA_NVCC_FLAGS_DEBUG ${CUDA_NVCC_FLAGS_DEBUG} "-G")

    file( GLOB  hdr *.hpp *.h )
#    file( GLOB  cu  *.cu)
#    SET (HW2_files main.cpp reference_calc.cpp compare.cpp)

    CUDA_ADD_EXECUTABLE(cudaFractal cudaFractal.cu ${hdr} ${cu})
    target_link_libraries(cudaFractal ${OPENGL_LIBRARIES} ${GLUT_LIBRARY} )
else(CUDA_FOUND)
    message("CUDA is not installed on this system.")
endif()
