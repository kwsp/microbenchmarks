vcpkg_from_github(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO kwsp/fftconv
    REF 704f6ebb9418cd6045dd856f90558cf9798674fb
    SHA512 94f52973cdff5f4fb479392b822d55781b144928e7dc9aa8fa78051351d2eeca6e351672238f87a2d1c84c04517e251f9683ebaa9fb6a3e7e022935e3d702a98
    HEAD_REF main
)

# Copy header only lib
file(INSTALL ${SOURCE_PATH}/fftconv_fftw/fftconv.hpp DESTINATION ${CURRENT_PACKAGES_DIR}/include)
file(INSTALL ${SOURCE_PATH}/fftconv_fftw/fftw.hpp DESTINATION ${CURRENT_PACKAGES_DIR}/include)

# Handle copyright
file(INSTALL "${SOURCE_PATH}/LICENSE" DESTINATION "${CURRENT_PACKAGES_DIR}/share/fftconv" RENAME copyright)
