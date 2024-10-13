# Define this to build the Flash Attention binding out of tree.
cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

file(GLOB FLASH_ATTENTION_SRCS
  "*.cpp"
)

file(GLOB FLASH_ATTENTION_CU
  "*.cu"
  "kernels/*.cu"
)

list(APPEND FLASH_ATTENTION_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/../../../../../../../third_party/cutlass/include)
list(APPEND FLASH_ATTENTION_SRCS ${FLASH_ATTENTION_CU})

set(FLASH_ATTENTION_SRCS ${FLASH_ATTENTION_SRCS} PARENT_SCOPE)
set(FLASH_ATTENTION_CU ${FLASH_ATTENTION_CU} PARENT_SCOPE)
set(FLASH_ATTENTION_INCLUDE ${FLASH_ATTENTION_INCLUDE} PARENT_SCOPE)

# add_library(pytorch_flash STATIC ${FLASH_ATTENTION_SRCS} )
# target_include_directories(pytorch_flash PRIVATE ${FLASH_ATTENTION_INCLUDE})

