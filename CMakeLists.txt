# To compile this project on Windows, you'll need to have:
#   nVidia GPU
#   CUDA Toolkit
#   Microsoft Visual C++ Build Tools with Windows SDK matching you Windows version (if you have Visual Studio installed you have that one already)
#
# You'll also need to adjust the variables here and in Compile.bat because most probably you'll have different paths than mine.
#
# Turn on CUSTOM_NVCC_BUILD when compiling inside CLion because it doesn't support Visual Studio compiler natively
# and it's the only Windows compiler that works with NVCC. This is a workaround to make VC++ compiler work in CLion.

cmake_minimum_required(VERSION 3.6)
project(ZZKG_CUDA)

set(CMAKE_CXX_STANDARD 14)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/out")
set(CUSTOM_NVCC_BUILD true)

#set(CUDA_TOOLKIT_ROOT_DIR "C:/ProgramData/NVIDIA GPU Computing Toolkit/v10.0")
#set(CUDA_NVCC_EXECUTABLE "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.0/bin")
#set(CUDA_INCLUDE_DIRS "C:/ProgramData/NVIDIA Corporation/CUDA Samples/v10.0/common/inc")
#set(CUDA_CUDART_LIBRARY "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.0/bin")
#

find_package(CUDA QUIET REQUIRED)

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

if (WIN32 AND CUSTOM_NVCC_BUILD)
    file(GLOB files "${CMAKE_SOURCE_DIR}/src/*.cu")
    foreach (file ${files})
        # Get only the filename without the absolute path
        get_filename_component(file ${file} NAME_WE)
        # This command is responsible for recompiling the .exe if a corresponding .cu file was modified
        add_custom_command(
                OUTPUT "${CMAKE_SOURCE_DIR}/out/${file}.exe"
                COMMAND cmd /C "${CMAKE_SOURCE_DIR}/Compile.bat ${file}.cu"
                DEPENDS "${CMAKE_SOURCE_DIR}/src/${file}.cu"
                #                DEPENDS "${CMAKE_SOURCE_DIR}/src/constants/*.h"
        )
        # Because this target depends on the .exe and CMake knows the upper command is responsible
        # for generating this .exe - it will be invoked before running the target. However the command
        # itself is configured to recompile the .exe only if the sorce file was modified. Clever, huh?
        message("Generating new target: ${file}")
        add_custom_target(
                ${file} ALL
                DEPENDS "${CMAKE_SOURCE_DIR}/out/${file}.exe"
        )
    endforeach ()

else ()
    file(GLOB files "${CMAKE_SOURCE_DIR}/src/*.cu")
    foreach (file ${files})
        # Get only the filename without the absolute path
        get_filename_component(file ${file} NAME_WE)
        message("Generating new target: ${file}")
        cuda_add_executable(
                ${file}
                src/${file}.cu
        )
    endforeach ()

endif ()

