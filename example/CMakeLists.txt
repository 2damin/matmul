add_executable(matmul
    main.cpp
    ${CMAKE_SOURCE_DIR}/src/matmul.hpp
    ${CMAKE_SOURCE_DIR}/src/matmul.cpp
    ${CMAKE_SOURCE_DIR}/src/matmul.cu)



target_link_libraries(matmul
    ${OpenCV_LIBS})