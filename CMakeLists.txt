cmake_minimum_required(VERSION 3.10)
project(Tensify)

set(CMAKE_CXX_STANDARD 17)

# Source files
set(SOURCE_FILES
    "src/pluginMain.cpp"
    "src/Tensify.cpp"
    "src/Tensify.h"
)

# Maya paths for Windows
set(MAYA_VERSION 2025)
set(MAYA_LOCATION "C:/Program Files/Autodesk/Maya2025")
set(MAYA_INCLUDE_DIR "${MAYA_LOCATION}/include")
set(MAYA_LIBRARY_DIR "${MAYA_LOCATION}/lib")

# Maya libraries to link
set(MAYA_LIBRARIES
    Foundation.lib
    OpenMaya.lib
    OpenMayaAnim.lib
    OpenMayaUI.lib
    OpenMayaRender.lib
)

# Print paths for debugging
message("#####################################################")
message("MAYA_LOCATION: ${MAYA_LOCATION}")
message("MAYA_INCLUDE_DIR: ${MAYA_INCLUDE_DIR}")
message("MAYA_LIBRARY_DIR: ${MAYA_LIBRARY_DIR}")
message("#####################################################")

# Setup build environment
include_directories(${MAYA_INCLUDE_DIR})
link_directories(${MAYA_LIBRARY_DIR})

# Build the plugin as shared library
add_library(${PROJECT_NAME} SHARED ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} ${MAYA_LIBRARIES})

# Plugin output settings
set_target_properties(${PROJECT_NAME} PROPERTIES
    OUTPUT_NAME "Tensify"
    PREFIX ""
    SUFFIX ".mll"
)

# Export entry points for Windows Maya plugin
if (WIN32)
    target_link_options(${PROJECT_NAME} PRIVATE "/export:initializePlugin" "/export:uninitializePlugin")
endif()

# Optional: install to plug-ins directory (used only if you run `cmake --install`)
install(TARGETS ${PROJECT_NAME} RUNTIME DESTINATION "plug-ins")
