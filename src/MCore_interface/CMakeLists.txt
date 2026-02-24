add_library(MCore_interface INTERFACE)
target_include_directories(MCore_interface INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/..)
target_link_libraries(MCore_interface INTERFACE MCore_util)
