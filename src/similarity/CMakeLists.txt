find_package(fmt CONFIG REQUIRED)
find_package(benchmark CONFIG REQUIRED)

find_package(IPP CONFIG)
if (IPP_FOUND)
  message("Found Intel IPP")
endif()

function(add_executable_script EXE_NAME)
  add_executable(${EXE_NAME} ${ARGN})
  set_target_properties(${EXE_NAME} PROPERTIES
      CXX_STANDARD 20
      CXX_EXTENSIONS OFF
  )

  if (APPLE)
    # Use Accelerate
    target_compile_definitions(${EXE_NAME} PRIVATE -DACCELERATE_NEW_LAPACK)
    target_link_libraries(${EXE_NAME} PRIVATE "-framework Accelerate")
  endif()

  target_link_libraries(${EXE_NAME} PRIVATE
    fmt::fmt
    armadillo
  )

  if (IPP_FOUND)
    target_link_libraries(${EXE_NAME} PRIVATE ${IPP_LIBRARIES})
    target_compile_definitions(${EXE_NAME} PRIVATE -DHAS_IPP)
  endif()

endfunction()


add_executable_script(similarity main.cpp)

add_executable_script(similarity_benchmarks benchmarks.cpp)
target_link_libraries(similarity_benchmarks PRIVATE benchmark::benchmark benchmark::benchmark_main)