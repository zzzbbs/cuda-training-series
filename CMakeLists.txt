cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(
  cuda-learning
  VERSION 0.1
  LANGUAGES CXX CUDA)

# option(RT_DRM_BUILD_GUI "Build rt-drm with GUI support" ON)
# option(RT_DRM_BUILD_TESTS "Build rt-drm test programs" ON)
# option(RT_DRM_USE_LIBTORCH "Build rt-drm with libtorch" OFF)

# set(TCMP_BUILD_TESTS OFF)

list(APPEND CUDA_NVCC_FLAGS "-Xcompiler=-fms-extensions")
list(APPEND CUDA_NVCC_FLAGS "--extended-lambda")
list(APPEND CUDA_NVCC_FLAGS "--expt-relaxed-constexpr")


set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

set(CMAKE_CUDA_RUNTIME_LIBRARY Shared)
set(CUDA_LINK_LIBRARIES_KEYWORD PUBLIC)
set(CMAKE_CUDA_ARCHITECTURES native)

set(CUDA_LIBS cublas cudart curand cufft)
set(NPP_LIBS
    nppc
    nppial
    nppicc
    nppidei
    nppif
    nppig
    nppim
    nppist
    nppisu
    nppitc
    npps)
set(CUDA_LIBRARIES ${CUDA_LIBS} ${NPP_LIBS})

list(APPEND CUDA_LEARN_LIBRARIES ${CUDA_LIBRARIES})

list(APPEND CUDA_LEARN_INCLUDES "${CMAKE_CURRENT_SOURCE_DIR}/include/")

file(GLOB_RECURSE CUDA_LEARN_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
     "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu"
     "${CMAKE_CURRENT_SOURCE_DIR}/src/hw1/*.cu")
message(${CUDA_LEARN_SOURCES})
# tcmp add_subdirectory(dependencies/tiny-cuda-mem-pool) list(APPEND
# RT_DRM_INCLUDES
# "${CMAKE_CURRENT_SOURCE_DIR}/dependencies/tiny-cuda-mem-pool/include")
# list(APPEND RT_DRM_LIBRARIES ${TCMP_LIBRARIES} tiny-cuda-mem-pool)

# eigen3 set(Eigen3_DIR "${CMAKE_CURRENT_SOURCE_DIR}/dependencies/eigen")
# set(EIGEN3_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third-party/eigen")
# set(EIGEN3_INCLUDE_DIR "${EIGEN3_DIR}")

# pinocchio find_package(urdfdom REQUIRED)
list(APPEND CUDA_LEARN_INCLUDES
     "/opt/galbot/thirdparty/gcc940-x86_64-ubuntu2004-gnu/include")
list(APPEND CUDA_LEARN_INCLUDES "/home/zbs/cuda/cuda-12.0/include")
if(LINUX)
  list(APPEND CUDA_LEARN_LIBRARIES pthread)
endif()

link_directories("/home/zbs/cuda/cuda-12.0/lib64")
link_directories("/opt/galbot/thirdparty/gcc940-x86_64-ubuntu2004-gnu/lib")
# set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/rt-drm/")
# set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_BINARY_DIR}/rt-drm")
# set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}/rt-drm")

# set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_DEBUG "${CMAKE_BINARY_DIR}/lib")
# set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}/lib")

# add_library(cuda_learning ${CUDA_LEARN_SOURCES})
# add_executable(cuda_learning ${CUDA_LEARN_SOURCES})
# # target_compile_options(cuda_learning PRIVATE "--keep")
# set_target_properties(cuda_learning PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON
#                                                CUDA_SEPARABLE_COMPILATION ON)
# target_include_directories(cuda_learning PUBLIC ${CUDA_LEARN_INCLUDES})
# target_link_libraries(cuda_learning PUBLIC ${CUDA_LEARN_LIBRARIES})
# target_compile_definitions(cuda_learning PUBLIC ${CUDA_LEARN_DEFINITIONS})
# target_compile_options(cuda_learning
#                        PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_NVCC_FLAGS}>)
# set_property(TARGET cuda_learning PROPERTY CXX_STANDARD 17)

# # The following code block is suggested to be used on Windows. According to #
# https://github.com/pytorch/pytorch/issues/25457, the DLLs need to be copied to
# # avoid memory errors. if(RT_DRM_USE_LIBTORCH) if(MSVC) file(GLOB TORCH_DLLS
# "${TORCH_INSTALL_PREFIX}/lib/*.dll") add_custom_command( TARGET rt-drm
# POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_if_different ${TORCH_DLLS}
# ${CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG}) endif(MSVC) endif()
add_subdirectory(src/hw1)
# Add test
if(CUDA_LEARN_BUILD_TESTS)
  add_subdirectory(tests)
endif()
