
cmake_minimum_required(VERSION 3.15 FATAL_ERROR)

# CMP0004: show an error if the path to a library contains whitespaces, instead of silently removing the whitespace:
cmake_policy(SET CMP0004 NEW)

project(VolumeFusion LANGUAGES CUDA CXX VERSION 1.0.0)

option(ENABLE_TEST "Build tests" ON)

list(APPEND CMAKE_MODULE_PATH
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules
    ${CMAKE_PREFIX_PATH})

# Configure dependencies:
set(VTK_COMPONENTS
    CommonCore
    CommonColor
    FiltersSources
    GUISupportQt
    InteractionStyle
    RenderingContextOpenGL2
    RenderingCore
    RenderingLabel
    RenderingFreeType
    RenderingGL2PSOpenGL2
    RenderingOpenGL2
)

# ------------------------------------------------------------------------
# dependencies

find_package(CUDA REQUIRED)

# BOOST
find_package(Boost 1.60.0 REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})

# JSON
find_package(nlohmann_json CONFIG REQUIRED)

# VTK
find_package(VTK COMPONENTS ${VTK_COMPONENTS} REQUIRED)
# include(${VTK_USE_FILE})

# Qt
set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_AUTOMOC ON)
SET(CMAKE_AUTOUIC ON)
# if(vtkGUISupportQt_LOADED OR TARGET VTK::GUISupportQt)
  # We have ui files, this will bring in the macro: qt5_wrap_ui
  find_package(Qt5Widgets REQUIRED)
  find_package(Qt5X11Extras REQUIRED)
#   if (Qt5_POSITION_INDEPENDENT_CODE)
    # set(CMAKE_POSITION_INDEPENDENT_CODE ON)
#   endif()
# else()
#   message(STATUS "VTK::GuiSupportQt not found.")
#   return()
# endif()

# Eigen
set(EIGEN_ROOT ${PROJECT_SOURCE_DIR}/third_party/include/eigen-master)
include_directories(${PROJECT_SOURCE_DIR}/third_party/include/eigen-master)
find_package (Eigen3 3.3 REQUIRED NO_MODULE PATHS ${PROJECT_SOURCE_DIR}/third_party/include/eigen-master)

# OpenCV
find_package(OpenCV 4.1.0 COMPONENTS calib3d core highgui imgproc video REQUIRED)
if(OPENCV_CORE_FOUND)
    include_directories(${OpenCV_INCLUDE_DIRS})
endif()

# Nanoflann, knnCuda, Sophus
include_directories("${PROJECT_SOURCE_DIR}/third_party/include/sophus"
                    "${PROJECT_SOURCE_DIR}/third_party/include/nanoflann"
                    "${PROJECT_SOURCE_DIR}/third_party/include/kNN-CUDA")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS NO)
# ------------------------------------------------------------------------
# C++ flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -funroll-loops -Wno-deprecated -pthread")

# CUDA flags
string(APPEND CMAKE_CUDA_FLAGS " -arch sm_61")
string(APPEND CMAKE_CUDA_FLAGS " --expt-relaxed-constexpr")
string(APPEND CMAKE_CUDA_FLAGS " --generate-line-info")
# string(APPEND CMAKE_CUDA_FLAGS " -O3")
# string(APPEND CMAKE_CUDA_FLAGS " --ptxas-options=-v")
string(APPEND CMAKE_CUDA_FLAGS " --compiler-options -Wall")
string(APPEND CMAKE_CUDA_FLAGS " --Wreorder")
string(APPEND CMAKE_CUDA_FLAGS " -Xcudafe --diag_suppress=esa_on_defaulted_function_ignored")
string(APPEND CMAKE_CUDA_FLAGS " -time nvcc_time.txt")
message(STATUS "CMAKE_CUDA_FLAGS ${CMAKE_CUDA_FLAGS}")
# set(CMAKE_CUDA_FLAGS ${CMAKE_CUDA_FLAGS};--device-c)
# set(CMAKE_CUDA_FLAGS ${CMAKE_CUDA_FLAGS};--use_fast_math)

# source directory
add_definitions(-DVOLUMEFUSION_SOURCE_DIR=${PROJECT_SOURCE_DIR})
add_definitions(-D_MWAITXINTRIN_H_INCLUDED -D_FORCE_INLINES -D__STRICT_ANSI__)

# ------------------------------------------------------------------------
# include directory
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

include(BDAHelperMethods)

# ------------------------------------------------------------------------
set(LIB_NAME "VolumeFusion")
set(APP_NAME "VolumeFusionGui")

set(SOURCES
    include/af/eigen_extension.h include/af/eigen_extension.cuh
    include/af/Helper.cuh src/Helper.cu
    include/af/Constants.h src/Constants.cpp
    include/af/Settings.h src/Settings.cpp
    include/af/CameraModel.h src/CameraModel.cpp
    include/af/dataset.h src/dataset.cpp
    include/af/KinectReader.h src/KinectReader.cpp
    include/af/AzureReader.h src/AzureReader.cpp
    include/af/DeviceImage.cuh
    include/af/DepthReader.cuh
    include/af/VertexCloud.h
    include/af/Mesh.h src/Mesh.cpp
    include/af/VertexManipulation.h src/VertexManipulation.cpp
    include/af/VertexManipulation.cuh src/VertexManipulation.cu
    include/af/MotionGraph.h src/MotionGraph.cpp
    include/af/GraphBuilder.h src/GraphBuilder.cpp
    include/af/TSDFVolume.h src/TSDFVolume.cpp
    include/af/TSDFIntegration.cuh src/TSDFIntegration.cu
    include/af/MarchingCubes.h src/MarchingCubes.cpp
    include/af/MarchingCubes.cuh src/MarchingCubes.cu
    include/af/ComputeCorrespondence.cuh src/ComputeCorrespondence.cu
    include/af/MotionOptimization.cuh src/MotionOptimization.cu
    include/af/LinearSolver.cuh src/LinearSolver.cu
    include/af/VolumeFusion.h src/VolumeFusion.cu
    include/af/DepthFrameComponent.cuh src/DepthFrameComponent.cu
    include/af/TSDFWarp.h src/TSDFWarp.cu
    src/VolumeFusionTest.cu
    include/af/WarpField.cuh src/WarpField.cu
    include/af/BilateralFilter.cuh src/BilateralFilter.cu
    include/af/Algorithms.cuh src/Algorithms.cu
    include/af/MeshRecontructionStream.cuh src/MeshRecontructionStream.cu
    include/af/PointCloudStream.cuh src/PointCloudStream.cu
    include/af/StreamBuffer.h
    include/af/CameraCalibration.h src/CameraCalibration.cpp
)

set(SOURCES_GUI
    src/ui/main.cpp
    include/af/ui/MainWindow.h src/ui/MainWindow.cpp
    include/af/ui/MainWindowUI.h
    include/af/ui/BufferVtkPoints.h src/ui/BufferVtkPoints.cu
    include/af/ui/BufferVector.h src/ui/BufferVector.cu
    include/af/ui/BufferObject.h src/ui/BufferObject.cu
    include/af/ui/warpMeshTmp.h src/ui/warpMeshTmp.cu
    include/af/ui/StepperWidget.h
    include/af/ui/ModifySignaler.h src/ui/ModifySignaler.cpp
    include/af/ui/ModifySignalerQt.h
    include/af/ui/PointsQWrap.h src/ui/PointsQWrap.cpp
    include/af/ui/VectorQWrap.h
)

set(LIBRARIES
    ${OpenCV_LIBS}
    ${CUDA_cusolver_LIBRARY}
    ${CUDA_cusparse_LIBRARY}
    ${CUDA_cublas_LIBRARY}
    ${CUDA_cublas_device_LIBRARY}
)

set(LIBRARIES_GUI
    Qt5::Core
    Qt5::Gui
    ${VTK_LIBRARIES}
    nlohmann_json
)

if ( TARGET Qt5::Core )
	get_property( core_options TARGET Qt5::Core PROPERTY INTERFACE_COMPILE_OPTIONS )
	string( REPLACE "-fPIC" "" new_core_options "${core_options}" )
	set_property( TARGET Qt5::Core PROPERTY INTERFACE_COMPILE_OPTIONS ${new_core_options} )
	set_property( TARGET Qt5::Core PROPERTY INTERFACE_POSITION_INDEPENDENT_CODE "ON" )
	set( CMAKE_CXX_COMPILE_OPTIONS_PIE "-fPIC" )
endif()

# Library
add_library(${LIB_NAME} ${SOURCES})
target_link_libraries(${LIB_NAME} ${LIBRARIES})
vtk_module_autoinit(
  TARGETS ${LIB_NAME}
  MODULES ${VTK_LIBRARIES}
)

# Application
add_executable(${APP_NAME} ${SOURCES_GUI})
target_link_libraries(${APP_NAME} ${LIB_NAME} ${LIBRARIES_GUI})
vtk_module_autoinit(
  TARGETS ${APP_NAME}
  MODULES ${VTK_LIBRARIES}
)





# -------------  TESTS

if(ENABLE_TEST)
    list(APPEND TESTFILES
        TSDFVolumeTest.cpp
        MotionGraphTest.cpp
        CudaTest.cu
        knnCUDATest.cu
        VertexManipulationTest.cu
        TSDFIntegrationTest.cu
        MarchingCubesTest.cu
        MotionOptimizationTest.cu
        TSDFWarpTest.cu
        BilateralFilterTest.cu
        VolumeFusionSpeedTest.cu
        )

    find_package(GTest 1.8.0 REQUIRED)
    enable_testing()

    foreach(TESTFILE ${TESTFILES})
        string(REPLACE "." ";" TESTNAME_EXT ${TESTFILE})
        list(GET TESTNAME_EXT 0 TESTNAME)
        list(GET TESTNAME_EXT 1 TESTEXT)

        add_executable(${TESTNAME}
            test/src/${TESTNAME}.${TESTEXT}
            test/src/GTestMain.cu
            src/Helper.cu)



        target_include_directories(${TESTNAME}
            PRIVATE
                ${CMAKE_CURRENT_SOURCE_DIR}/include
                ${CMAKE_CURRENT_SOURCE_DIR}/test/src
                ${CMAKE_CURRENT_SOURCE_DIR}/src
                ${CMAKE_CURRENT_BINARY_DIR})

        target_compile_options(${TESTNAME}
            PRIVATE
                $<$<CXX_COMPILER_ID:GNU>:-Wall;-pedantic;-Wextra>)

        target_link_libraries(${TESTNAME} ${LIB_NAME}  GTest::GTest ${CUDA_cusolver_LIBRARY} ${CUDA_cusparse_LIBRARY})

        add_test(NAME ${TESTNAME} COMMAND ${TESTNAME})
        set_tests_properties(${TESTNAME} PROPERTIES TIMEOUT 1800)
    endforeach()

    # file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/test")
    # file(COPY "test/resources/" DESTINATION "${CMAKE_CURRENT_BINARY_DIR}/test/resources/")
endif()


# Add uninstall target
configure_file(cmake/CMakeUninstall.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/CMakeUninstall.cmake IMMEDIATE @ONLY)

add_custom_target(uninstall
    COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_BINARY_DIR}/CMakeUninstall.cmake)

# Export the package for use from the build tree
export(TARGETS ${PROJECT_NAME} NAMESPACE BDA:: FILE cmake/${PROJECT_NAME}Targets.cmake)
export(PACKAGE ${PROJECT_NAME})

include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    cmake/${PROJECT_NAME}ConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY AnyNewerVersion)

configure_package_config_file(
    cmake/${PROJECT_NAME}Config.cmake.in
    cmake/${PROJECT_NAME}Config.cmake
    INSTALL_DESTINATION lib/cmake/)

include(GenerateExportHeader)
generate_export_header(${PROJECT_NAME} EXPORT_FILE_NAME af/volumefusion_export.h)

install(TARGETS ${PROJECT_NAME}
    EXPORT ${PROJECT_NAME}Targets
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
    INCLUDES DESTINATION include)
install(DIRECTORY include/af DESTINATION include)

install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/af/volumefusion_export.h DESTINATION include/af/)

install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/cmake/${PROJECT_NAME}Config.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/cmake/${PROJECT_NAME}ConfigVersion.cmake
    DESTINATION lib/cmake/)

install(EXPORT ${PROJECT_NAME}Targets
    NAMESPACE BDA:: DESTINATION lib/cmake/)
