vcpkg_from_github(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO kwsp/fftconv
    REF 0d9b3825215408a8e153552bff070fac28569d1f
    SHA512 8e97b842f518c9dc17cb77ba9cf7606e6cd2ea210cb7b46ae411851ed66cf33692b006ecf601085bccf73c9a5a08bd5a1da2a28d03be5a3802102f09d55cd4c2
    HEAD_REF main
)

# Copy header only lib
file(INSTALL ${SOURCE_PATH}/fftconv_fftw/fftconv.hpp DESTINATION ${CURRENT_PACKAGES_DIR}/include)
file(INSTALL ${SOURCE_PATH}/fftconv_fftw/fftw.hpp DESTINATION ${CURRENT_PACKAGES_DIR}/include)

# Handle copyright
file(INSTALL "${SOURCE_PATH}/LICENSE" DESTINATION "${CURRENT_PACKAGES_DIR}/share/fftconv" RENAME copyright)
