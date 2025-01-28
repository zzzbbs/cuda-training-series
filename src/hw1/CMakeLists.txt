add_executable(cuda_learning_hw1_hello ${CMAKE_CURRENT_SOURCE_DIR}/hello.cu)
# target_compile_options(cuda_learning PRIVATE "--keep")
set_target_properties(cuda_learning_hw1_hello PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON
                                               CUDA_SEPARABLE_COMPILATION ON)
target_include_directories(cuda_learning_hw1_hello PUBLIC ${CUDA_LEARN_INCLUDES})
target_link_libraries(cuda_learning_hw1_hello PUBLIC ${CUDA_LEARN_LIBRARIES})
target_compile_definitions(cuda_learning_hw1_hello PUBLIC ${CUDA_LEARN_DEFINITIONS})
target_compile_options(cuda_learning_hw1_hello
                       PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_NVCC_FLAGS}>)

add_executable(cuda_learning_hw1_matrix_mul ${CMAKE_CURRENT_SOURCE_DIR}/matrix_mul.cu)
# target_compile_options(cuda_learning PRIVATE "--keep")
set_target_properties(cuda_learning_hw1_matrix_mul PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON
                                               CUDA_SEPARABLE_COMPILATION ON)
target_include_directories(cuda_learning_hw1_matrix_mul PUBLIC ${CUDA_LEARN_INCLUDES})
target_link_libraries(cuda_learning_hw1_matrix_mul PUBLIC ${CUDA_LEARN_LIBRARIES})
target_compile_definitions(cuda_learning_hw1_matrix_mul PUBLIC ${CUDA_LEARN_DEFINITIONS})
target_compile_options(cuda_learning_hw1_matrix_mul
                       PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_NVCC_FLAGS}>)

add_executable(cuda_learning_hw1_vector_add ${CMAKE_CURRENT_SOURCE_DIR}/vector_add.cu)
# target_compile_options(cuda_learning PRIVATE "--keep")
set_target_properties(cuda_learning_hw1_vector_add PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON
                                               CUDA_SEPARABLE_COMPILATION ON)
target_include_directories(cuda_learning_hw1_vector_add PUBLIC ${CUDA_LEARN_INCLUDES})
target_link_libraries(cuda_learning_hw1_vector_add PUBLIC ${CUDA_LEARN_LIBRARIES})
target_compile_definitions(cuda_learning_hw1_vector_add PUBLIC ${CUDA_LEARN_DEFINITIONS})
target_compile_options(cuda_learning_hw1_vector_add
                       PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_NVCC_FLAGS}>)