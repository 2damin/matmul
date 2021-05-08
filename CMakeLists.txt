cmake_minimum_required(VERSION 3.10)

PROJECT(matmul)

#######################
# CUDA SETTING

find_package(CUDA 11.1)

include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(CUDA_ARCHITECTURES "Auto" CACHE STRING "\"Auto\" detects local machine GPU compute arch at runtime, \"Common\" and \"All\" cover common and entire subsets of architectures, \"Names\" is a list of architectures to enable by name, \"Numbers\" is a list of compute capabilities (version number) to enable")
    set_property(CACHE CUDA_ARCHITECTURES PROPERTY STRINGS "Auto" "Common" "All" "Kepler Maxwell Kepler+Tegra Maxwell+Tegra Pascal" "5.0 7.5")
    cuda_select_nvcc_arch_flags(CUDA_ARCH_FLAGS ${CUDA_ARCHITECTURES})
    if (CMAKE_VERSION VERSION_GREATER_EQUAL "3.17")
        if(BUILD_SHARED_LIBS)
            set(CMAKE_CUDA_RUNTIME_LIBRARY "Shared")
        else()
            set(CMAKE_CUDA_RUNTIME_LIBRARY "Static")
        endif()
    endif()
else()
    set(ENABLE_CUDA "FALSE" CACHE BOOL "Enable CUDA build" FORCE)
    message(SEND_ERROR "CMake cannot locate a CUDA compiler")
endif()


if(CMAKE_CUDA_COMPILER AND ENABLE_CUDA)
  set(CUDA_ARCHITECTURES "Auto" CACHE STRING "\"Auto\" detects local machine GPU compute arch at runtime, \"Common\" and \"All\" cover common and entire subsets of architectures, \"Names\" is a list of architectures to enable by name, \"Numbers\" is a list of compute capabilities (version number) to enable")
  set_property(CACHE CUDA_ARCHITECTURES PROPERTY STRINGS "Auto" "Common" "All" "Kepler Maxwell Kepler+Tegra Maxwell+Tegra Pascal" "5.0 7.5")
  enable_language(CUDA)
  if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS "9.0")
    message(STATUS "Unsupported CUDA version, please upgrade to CUDA 9+. Disabling CUDA support")
    set(ENABLE_CUDA "FALSE" CACHE BOOL "Enable CUDA support" FORCE)
  else()
    find_package(CUDA REQUIRED)
    cuda_select_nvcc_arch_flags(CUDA_ARCH_FLAGS ${CUDA_ARCHITECTURES})
    message(STATUS "Building with CUDA flags: " "${CUDA_ARCH_FLAGS}")
    if (NOT "arch=compute_70,code=sm_70" IN_LIST CUDA_ARCH_FLAGS AND NOT "arch=compute_72,code=sm_72" IN_LIST CUDA_ARCH_FLAGS AND NOT "arch=compute_75,code=sm_75" IN_LIST CUDA_ARCH_FLAGS AND NOT "arch=compute_80,code=sm_80" IN_LIST CUDA_ARCH_FLAGS)
      set(ENABLE_CUDNN_HALF "FALSE" CACHE BOOL "Enable CUDNN Half precision" FORCE)
      message(STATUS "Your setup does not supports half precision (it requires CC >= 7.0)")
    else()
      message(STATUS "Your setup supports half precision (it requires CC >= 7.0)")
    endif()
  endif()
  if (CMAKE_VERSION VERSION_GREATER_EQUAL "3.17")
    if(BUILD_SHARED_LIBS)
      set(CMAKE_CUDA_RUNTIME_LIBRARY "Shared")
    else()
      set(CMAKE_CUDA_RUNTIME_LIBRARY "Static")
    endif()
  endif()
  string (REPLACE ";" " " CUDA_ARCH_FLAGS_SPACE_SEPARATED "${CUDA_ARCH_FLAGS}")
  set(CMAKE_CUDA_FLAGS "${CUDA_ARCH_FLAGS_SPACE_SEPARATED} ${CUDA_HOST_COMPILER_FLAGS} ${CMAKE_CUDA_FLAGS}")
  message(STATUS "CMAKE_CUDA_FLAGS: ${CMAKE_CUDA_FLAGS}")
else()
  set(ENABLE_CUDA "FALSE" CACHE BOOL "Enable CUDA support" FORCE)
endif()

#######################
# BUILD SETTING
if (WIN32)
    if (MSVC)
        # Standard:          /DWIN32 /D_WINDOWS /W3 /Zm1000 /EHsc /GR
        # Reference: https://msdn.microsoft.com/en-us/library/fwkeyyhe.aspx
        # WIN32 --> Win32 console application
        # Wx  --> Warning level
        # Zi  --> Produces a program database (PDB) for complete debug information
        # EHsc --> Exception handling model (catches C++ exceptions only)
        # GR(-) --> Enables / Disables Run-Time Type Information
        # MP  --> Multi-processes compilation
        # openmp --> Enables openmp
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP /bigobj")
    endif (MSVC)
endif (WIN32)

IF(NOT MSVC)
    set(DEFAULT_BUILD_TYPE "Release")

    if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
        message(STATUS "Setting build type to '${DEFAULT_BUILD_TYPE}' since none specified.")
        set(CMAKE_BUILD_TYPE "${DEFAULT_BUILD_TYPE}" CACHE STRING "Choose the type of build" FORCE)
        set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
    endif()
endif()


set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
ADD_DEFINITIONS(-D_SILENCE_ALL_CXX17_DEPRECATION_WARNINGS)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib/${CMAKE_SYSTEM_NAME}/${CMAKE_SYSTEM_PROCESSOR}/${CMAKE_BUILD_TYPE})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib/${CMAKE_SYSTEM_NAME}/${CMAKE_SYSTEM_PROCESSOR}/${CMAKE_BUILD_TYPE})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin/)

#######################
# opencv setting

find_package(OpenCV REQUIRED)

if( OpenCV_FOUND )
    message(STATUS "OpenCV IncludeDir ${OpenCV_INCLUDE_DIRS}")
    message(STATUS "OpenCV Libs ${OpenCV_LIBS}")
endif()

include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/src
    ${CUDA_INCLUDE_DIRS}
)

link_directories(
    ${OpenCV_LIBS}
)

add_subdirectory(example)
