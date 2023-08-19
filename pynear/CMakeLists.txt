cmake_minimum_required(VERSION 3.10)


project(VPTree LANGUAGES CXX VERSION 1.0.0)
include(GNUInstallDirs)

set(CMAKE_CXX_STANDARD 17)

find_package(PythonLibs REQUIRED)
find_package(pybind11 REQUIRED)
find_package(OpenMP)

set(DEFAULT_BUILD_TYPE "Debug")

if(WIN32)
    SET(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} /Wall /arch:AVX /openmp")
    SET(CMAKE_EXE_LINKER_FLAGS  "${CMAKE_EXE_LINKER_FLAGS} /LTCG")
else()
    SET(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -flto -Wall -march=native -mavx -fopenmp")
    if(APPLE)
        SET(CMAKE_EXE_LINKER_FLAGS  "${CMAKE_EXE_LINKER_FLAGS} -fopenmp -lomp")
    else()
        SET(CMAKE_EXE_LINKER_FLAGS  "${CMAKE_EXE_LINKER_FLAGS} -fopenmp -lgomp")
    endif()
endif(WIN32)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to '${DEFAULT_BUILD_TYPE}' as none was specified.")
  set(CMAKE_BUILD_TYPE "${DEFAULT_BUILD_TYPE}" CACHE STRING "Choose the type of build." FORCE)
  # Set the possible values of build type for cmake-gui
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

enable_testing()
add_library(${PROJECT_NAME} INTERFACE)

target_include_directories(${PROJECT_NAME} INTERFACE
    ${pybind11_INCLUDE_DIR}
    ${PYTHON_INCLUDE_DIRS}
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

set_target_properties(${PROJECT_NAME} PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION 1)
set_target_properties(${PROJECT_NAME} PROPERTIES PUBLIC_HEADER "${CMAKE_CURRENT_SOURCE_DIR}/include/VPTree.hpp ${CMAKE_CURRENT_SOURCE_DIR}/include/DistanceFunctions.hpp")

set(CONFIG_NAME ${PROJECT_NAME}Config)

pybind11_add_module(pyvptree MODULE src/PythonBindings.cpp )

target_link_libraries(pyvptree PUBLIC OpenMP::OpenMP_CXX)
target_include_directories(pyvptree PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

execute_process(
  COMMAND "${PYTHON_EXECUTABLE}" -c "if True:
    from distutils import sysconfig as sc
    print(sc.get_python_lib(prefix='', plat_specific=True))"
  OUTPUT_VARIABLE PYTHON_SITE
  OUTPUT_STRIP_TRAILING_WHITESPACE)

install(TARGETS ${PROJECT_NAME} pyvptree COMPONENT python EXPORT ${CONFIG_NAME}
    ARCHIVE DESTINATION ${PYTHON_SITE}
    LIBRARY DESTINATION ${PYTHON_SITE}
    RUNTIME DESTINATION ${PYTHON_SITE}
    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME}
)

# Install the config file into the project name within share so it can be found by other cmake projects
install(EXPORT ${CONFIG_NAME} DESTINATION share/${PROJECT_NAME}/cmake)

export(TARGETS ${PROJECT_NAME} FILE ${CONFIG_NAME}.cmake)
add_subdirectory(tests)
