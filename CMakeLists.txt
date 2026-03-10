cmake_minimum_required(VERSION 3.20)
project(vivid-3d VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- Locate vivid source and build trees ---
# VIVID_SRC_DIR: path to vivid source tree (for operator_api headers, linmath.h)
# VIVID_BUILD_DIR: path to vivid build tree (for WebGPU)
if(NOT DEFINED VIVID_SRC_DIR)
    # Default: sibling directory
    set(VIVID_SRC_DIR "${CMAKE_SOURCE_DIR}/../vivid")
endif()
if(NOT DEFINED VIVID_BUILD_DIR)
    set(VIVID_BUILD_DIR "${VIVID_SRC_DIR}/build")
endif()

# Build package test executables (disabled for install/rebuild by default)
option(VIVID_PACKAGE_BUILD_TESTS "Build vivid-3d package tests" OFF)

# --- Operator API (header-only) ---
add_library(vivid_operator_api INTERFACE)
target_include_directories(vivid_operator_api INTERFACE
    ${VIVID_SRC_DIR}/src            # operator_api/*.h
    ${VIVID_SRC_DIR}/deps/glfw/deps # linmath.h
    ${CMAKE_SOURCE_DIR}/include     # vivid-3d's gpu_3d.h
)

# --- WebGPU discovery ---
# Prefer the host runtime's libwgpu_native to avoid loading a second WebGPU
# runtime in package plugins.
set(_wgpu_include "")
set(_wgpu_lib_dir "")

if(EXISTS "${VIVID_SRC_DIR}/include/webgpu/webgpu.h")
    set(_wgpu_include "${VIVID_SRC_DIR}/include")
endif()

if(EXISTS "${VIVID_BUILD_DIR}/vivid.app/Contents/MacOS/libwgpu_native.dylib")
    set(_wgpu_lib_dir "${VIVID_BUILD_DIR}/vivid.app/Contents/MacOS")
elseif(EXISTS "${VIVID_BUILD_DIR}/libwgpu_native.dylib")
    set(_wgpu_lib_dir "${VIVID_BUILD_DIR}")
endif()

# Fallback for unusual build trees
if(_wgpu_include STREQUAL "" OR _wgpu_lib_dir STREQUAL "")
    set(_deps_dir "${VIVID_BUILD_DIR}/_deps")
    if(IS_DIRECTORY "${_deps_dir}")
        file(GLOB _dep_entries "${_deps_dir}/*")
        foreach(_entry ${_dep_entries})
            get_filename_component(_entry_name "${_entry}" NAME)
            if(_entry_name MATCHES "wgpu.*-src")
                if(_wgpu_include STREQUAL "" AND EXISTS "${_entry}/include/webgpu/webgpu.h")
                    set(_wgpu_include "${_entry}/include")
                endif()
                if(_wgpu_lib_dir STREQUAL "" AND IS_DIRECTORY "${_entry}/lib")
                    set(_wgpu_lib_dir "${_entry}/lib")
                endif()
            endif()
        endforeach()
    endif()
endif()

add_library(webgpu INTERFACE)
if(NOT _wgpu_include STREQUAL "")
    target_include_directories(webgpu INTERFACE "${_wgpu_include}")
endif()
if(NOT _wgpu_lib_dir STREQUAL "")
    if(EXISTS "${_wgpu_lib_dir}/libwgpu_native.dylib")
        set(WGPU_NATIVE_LIB "${_wgpu_lib_dir}/libwgpu_native.dylib")
    else()
        unset(WGPU_NATIVE_LIB CACHE)
        find_library(WGPU_NATIVE_LIB wgpu_native PATHS "${_wgpu_lib_dir}" NO_DEFAULT_PATH)
    endif()
    if(WGPU_NATIVE_LIB)
        target_link_libraries(webgpu INTERFACE "${WGPU_NATIVE_LIB}")
    endif()
endif()
# --- Platform plugin suffix ---
if(APPLE)
    set(VIVID_PLUGIN_SUFFIX ".dylib")
elseif(WIN32)
    set(VIVID_PLUGIN_SUFFIX ".dll")
else()
    set(VIVID_PLUGIN_SUFFIX ".so")
endif()

# --- Operator plugin helper ---
function(add_3d_operator name source)
    cmake_parse_arguments(ARG "" "" "EXTRA_LIBS;EXTRA_INCLUDES" ${ARGN})
    add_library(${name} MODULE ${source})
    if(APPLE)
        set(_target_suffix ".dylib")
    elseif(WIN32)
        set(_target_suffix ".dll")
    else()
        set(_target_suffix ".so")
    endif()
    set_target_properties(${name} PROPERTIES
        PREFIX ""
        OUTPUT_NAME "${name}"
        SUFFIX "${_target_suffix}"
    )
    target_link_libraries(${name} PRIVATE vivid_operator_api webgpu ${ARG_EXTRA_LIBS})
    if(ARG_EXTRA_INCLUDES)
        target_include_directories(${name} PRIVATE ${ARG_EXTRA_INCLUDES})
    endif()
endfunction()
# --- Tier 1: raw geometry generators (GPU_MESH output, no scene overhead) ---
add_3d_operator(grid             operators/gpu/grid/grid.cpp)
add_3d_operator(sphere           operators/gpu/sphere/sphere.cpp)
add_3d_operator(box              operators/gpu/box/box.cpp)
add_3d_operator(point_cloud      operators/gpu/point_cloud/point_cloud.cpp)
add_3d_operator(instanced_render operators/gpu/instanced_render/instanced_render.cpp)

# --- 3D operator plugins ---
add_3d_operator(render_3d     operators/gpu/render_3d/render_3d.cpp)
add_3d_operator(shape3d       operators/gpu/shape3d/shape3d.cpp)
add_3d_operator(transform3d   operators/gpu/transform3d/transform3d.cpp)
add_3d_operator(scene_merge   operators/gpu/scene_merge/scene_merge.cpp)
add_3d_operator(light3d       operators/gpu/light3d/light3d.cpp)
add_3d_operator(mesh_import   operators/gpu/mesh_import/mesh_import.cpp
    EXTRA_INCLUDES
        ${CMAKE_SOURCE_DIR}/deps/cgltf
        ${CMAKE_SOURCE_DIR}/deps/tinyobjloader
        ${VIVID_SRC_DIR}/deps/stb)
add_3d_operator(deformer      operators/gpu/deformer/deformer.cpp)
add_3d_operator(instancer3d   operators/gpu/instancer3d/instancer3d.cpp)
add_3d_operator(particles3d   operators/gpu/particles3d/particles3d.cpp)
add_3d_operator(sdf3d         operators/gpu/sdf3d/sdf3d.cpp)
add_3d_operator(material3d    operators/gpu/material3d/material3d.cpp)
add_3d_operator(ssao3d        operators/gpu/ssao3d/ssao3d.cpp)
add_3d_operator(dof3d         operators/gpu/dof3d/dof3d.cpp)
add_3d_operator(environment3d operators/gpu/environment3d/environment3d.cpp)
add_3d_operator(orbit_camera  operators/control/orbit_camera/orbit_camera.cpp)

# --- Manifold (CSG Boolean library) ---
include(FetchContent)
FetchContent_Declare(
    manifold
    GIT_REPOSITORY https://github.com/elalish/manifold.git
    GIT_TAG        v3.0.1
    GIT_SHALLOW    TRUE
)
set(MANIFOLD_TEST OFF CACHE BOOL "" FORCE)
set(MANIFOLD_CROSS_SECTION OFF CACHE BOOL "" FORCE)
set(MANIFOLD_EXPORT OFF CACHE BOOL "" FORCE)
set(MANIFOLD_PYBIND OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(manifold)

add_3d_operator(boolean3d operators/gpu/boolean3d/boolean3d.cpp EXTRA_LIBS manifold)

# --- Operators meta-target (for smoke CI: builds only dylibs, skips test executables) ---
add_custom_target(operators DEPENDS
    grid sphere box point_cloud instanced_render
    render_3d shape3d transform3d scene_merge light3d mesh_import
    deformer instancer3d particles3d sdf3d material3d ssao3d dof3d
    environment3d boolean3d orbit_camera)

# --- Tests ---
if(VIVID_PACKAGE_BUILD_TESTS)
enable_testing()

# Locate vivid runtime sources needed for scheduler-based tests
set(VIVID_RUNTIME_SOURCES
    ${VIVID_SRC_DIR}/src/runtime/operator_loader.cpp
    ${VIVID_SRC_DIR}/src/runtime/operator_registry.cpp
    ${VIVID_SRC_DIR}/src/runtime/wgsl_header_parser.cpp
    ${VIVID_SRC_DIR}/src/runtime/graph.cpp
    ${VIVID_SRC_DIR}/src/runtime/scheduler.cpp
)

# Find yyjson in the vivid build tree
set(YYJSON_INCLUDE_DIR "${VIVID_SRC_DIR}/deps/yyjson")
find_library(YYJSON_LIB yyjson PATHS "${VIVID_BUILD_DIR}" NO_DEFAULT_PATH)
if(NOT YYJSON_LIB)
    # Build yyjson from vivid source
    add_library(yyjson STATIC ${VIVID_SRC_DIR}/deps/yyjson/yyjson.c)
    target_include_directories(yyjson PUBLIC ${YYJSON_INCLUDE_DIR})
    set(YYJSON_LIB yyjson)
endif()

# Helper for 3D test executables
function(add_3d_test name source)
    cmake_parse_arguments(ARG "" "" "DEPENDS;EXTRA_INCLUDES" ${ARGN})
    add_executable(${name} ${source} ${VIVID_RUNTIME_SOURCES})
    target_include_directories(${name} PRIVATE
        ${VIVID_SRC_DIR}/src
        ${CMAKE_SOURCE_DIR}/include
        ${YYJSON_INCLUDE_DIR}
        ${ARG_EXTRA_INCLUDES})
    target_link_libraries(${name} PRIVATE vivid_operator_api ${YYJSON_LIB} webgpu)
    if(ARG_DEPENDS)
        add_dependencies(${name} ${ARG_DEPENDS})
    endif()
    add_test(NAME ${name} COMMAND ${name} WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
endfunction()

# Unit tests
add_executable(test_gpu_3d tests/test_gpu_3d.cpp)
target_include_directories(test_gpu_3d PRIVATE
    ${VIVID_SRC_DIR}/src
    ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(test_gpu_3d PRIVATE vivid_operator_api webgpu)
add_test(NAME test_gpu_3d COMMAND test_gpu_3d)

add_3d_test(test_render_3d    tests/test_render_3d.cpp    DEPENDS render_3d)
add_3d_test(test_shape3d      tests/test_shape3d.cpp      DEPENDS shape3d render_3d)
add_3d_test(test_material3d   tests/test_material3d.cpp   DEPENDS shape3d material3d render_3d)
add_3d_test(test_scene3d      tests/test_scene3d.cpp      DEPENDS shape3d render_3d transform3d scene_merge light3d)
add_3d_test(test_shadow3d     tests/test_shadow3d.cpp     DEPENDS shape3d render_3d transform3d scene_merge light3d)
add_3d_test(test_depth_output tests/test_depth_output.cpp DEPENDS shape3d render_3d transform3d scene_merge)
add_3d_test(test_ssao3d       tests/test_ssao3d.cpp       DEPENDS ssao3d shape3d render_3d transform3d scene_merge)
add_3d_test(test_dof3d        tests/test_dof3d.cpp        DEPENDS dof3d shape3d render_3d transform3d scene_merge)
add_3d_test(test_environment3d tests/test_environment3d.cpp DEPENDS environment3d shape3d material3d render_3d scene_merge light3d)
add_3d_test(test_mesh_import  tests/test_mesh_import.cpp  DEPENDS mesh_import render_3d
    EXTRA_INCLUDES ${CMAKE_SOURCE_DIR}/deps/tinyobjloader)
add_3d_test(test_deformer     tests/test_deformer.cpp     DEPENDS shape3d deformer render_3d)
add_3d_test(test_instancer3d  tests/test_instancer3d.cpp  DEPENDS shape3d instancer3d render_3d)
add_3d_test(test_boolean3d    tests/test_boolean3d.cpp    DEPENDS shape3d boolean3d render_3d)
add_3d_test(test_particles3d  tests/test_particles3d.cpp  DEPENDS particles3d render_3d)
add_3d_test(test_sdf3d        tests/test_sdf3d.cpp        DEPENDS sdf3d render_3d shape3d scene_merge)

# Copy test data
configure_file(tests/data/tetrahedron.obj ${CMAKE_BINARY_DIR}/tests/data/tetrahedron.obj COPYONLY)

endif()

# Copy demo graphs
file(GLOB _3D_GRAPHS graphs/*.json)
foreach(g ${_3D_GRAPHS})
    get_filename_component(name ${g} NAME)
    configure_file(${g} ${CMAKE_BINARY_DIR}/graphs/${name} COPYONLY)
endforeach()
