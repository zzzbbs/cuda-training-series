add_executable(cuda_learning_hw2_matrix_mul_shared
               ${CMAKE_CURRENT_SOURCE_DIR}/matrix_mul_shared.cu)
# target_compile_options(cuda_learning PRIVATE "--keep")
set_target_properties(
  cuda_learning_hw2_matrix_mul_shared PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON
                                                 CUDA_SEPARABLE_COMPILATION ON)
target_include_directories(cuda_learning_hw2_matrix_mul_shared
                           PUBLIC ${CUDA_LEARN_INCLUDES})
target_link_libraries(cuda_learning_hw2_matrix_mul_shared
                      PUBLIC ${CUDA_LEARN_LIBRARIES})
target_compile_definitions(cuda_learning_hw2_matrix_mul_shared
                           PUBLIC ${CUDA_LEARN_DEFINITIONS})
target_compile_options(cuda_learning_hw2_matrix_mul_shared
                       PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_NVCC_FLAGS}>)

add_executable(cuda_learning_hw2_stencil_1d
               ${CMAKE_CURRENT_SOURCE_DIR}/stencil_1d.cu)
# target_compile_options(cuda_learning PRIVATE "--keep")
set_target_properties(
    cuda_learning_hw2_stencil_1d PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON
                                                 CUDA_SEPARABLE_COMPILATION ON)
target_include_directories(cuda_learning_hw2_stencil_1d
                           PUBLIC ${CUDA_LEARN_INCLUDES})
target_link_libraries(cuda_learning_hw2_stencil_1d
                      PUBLIC ${CUDA_LEARN_LIBRARIES})
target_compile_definitions(cuda_learning_hw2_stencil_1d
                           PUBLIC ${CUDA_LEARN_DEFINITIONS})
target_compile_options(cuda_learning_hw2_stencil_1d
                       PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_NVCC_FLAGS}>)
