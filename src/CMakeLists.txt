set(EXE_NAME main)

add_executable(${EXE_NAME}
  main.cpp
)


set_target_properties(${EXE_NAME} PROPERTIES
    CXX_STANDARD 20
    CXX_EXTENSIONS OFF
    # INTERPROCEDURAL_OPTIMIZATION TRUE
)
