
find_package (Eigen3 3.3 REQUIRED NO_MODULE)

include_directories(
    ${CMAKE_SOURCE_DIR}/
    ${CMAKE_CURRENT_SOURCE_DIR}/
)

    
link_directories(
    ${CoreLibraries_BINARY_DIR}/Utilities 
    ${BOOST_LIBRARY}
    /usr/lib
    /usr/local/lib
    ) 

# Set the compiler flags
add_definitions("-Wall" -O3)

set(ils_SRCS
    main.cpp
)

set(ils_LIBS
    m 
    Eigen3::Eigen
)

# Main Program
add_executable(integer_least_squares ${ils_SRCS})
target_link_libraries(integer_least_squares ${ils_LIBS})

INSTALL(TARGETS integer_least_squares DESTINATION bin)


