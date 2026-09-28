cmake_minimum_required(VERSION 3.15)
project(acyclic_join_sgx C CXX)

option(ENABLE_MEMORY_PROFILE "Track enclave C++ heap allocations for --memory-profile" OFF)
set(JFYAN_MAX_MATERIALIZED_ROWS "200000000" CACHE STRING
    "Public maximum output rows (1..268435455)")
if(NOT JFYAN_MAX_MATERIALIZED_ROWS MATCHES "^[1-9][0-9]*$" OR
   JFYAN_MAX_MATERIALIZED_ROWS GREATER 268435455)
  message(FATAL_ERROR "JFYAN_MAX_MATERIALIZED_ROWS must be an integer in [1, 268435455]")
endif()

if(NOT DEFINED SGX_SDK)
  set(SGX_SDK /opt/intel/sgxsdk)
endif()

if(NOT DEFINED SGX_MODE)
  set(SGX_MODE HW)
endif()
if(SGX_MODE STREQUAL "HW")
  set(SGX_TRTS_LIBRARY sgx_trts)
  set(SGX_TSERVICE_LIBRARY sgx_tservice)
elseif(SGX_MODE STREQUAL "SIM")
  set(SGX_TRTS_LIBRARY sgx_trts_sim)
  set(SGX_TSERVICE_LIBRARY sgx_tservice_sim)
else()
  message(FATAL_ERROR "SGX_MODE must be HW or SIM")
endif()

if(NOT DEFINED SGX_DEBUG)
  set(SGX_DEBUG 1)
endif()

message(STATUS "SGX_SDK = ${SGX_SDK}")
message(STATUS "SGX_MODE = ${SGX_MODE}")
message(STATUS "SGX_DEBUG = ${SGX_DEBUG}")
message(STATUS "ENABLE_MEMORY_PROFILE = ${ENABLE_MEMORY_PROFILE}")
message(STATUS "JFYAN_MAX_MATERIALIZED_ROWS = ${JFYAN_MAX_MATERIALIZED_ROWS}")

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")

set(SGX_EDGER8R ${SGX_SDK}/bin/x64/sgx_edger8r)
set(SGX_SIGN    ${SGX_SDK}/bin/x64/sgx_sign)
set(SGX_COMMON_INC ${SGX_SDK}/include)
set(SGX_LIB_DIR    ${SGX_SDK}/lib64)

include_directories(${SGX_COMMON_INC})

if(SGX_DEBUG EQUAL 1)
  add_compile_definitions(DEBUG)
  set(SGX_DEBUG_FLAG SGX_DEBUG_FLAG)
else()
  set(SGX_DEBUG_FLAG 0)
endif()

add_subdirectory(Enclave)
add_subdirectory(App)
