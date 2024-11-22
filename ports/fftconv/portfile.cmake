vcpkg_from_github(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO kwsp/fftconv
    REF 066a11e
    SHA512 96c77a53e4612e38bba6133df70aa8a9f8c9baffe4806259982410f2d68b8ba58ae43448ce5deee646280fff96f3ca0b4185a0566c43463a375860c7fa2abc0c
    HEAD_REF main
)

# Copy header only lib
file(INSTALL ${SOURCE_PATH}/fftconv_fftw/fftconv.hpp DESTINATION ${CURRENT_PACKAGES_DIR}/include)
file(INSTALL ${SOURCE_PATH}/fftconv_fftw/fftw.hpp DESTINATION ${CURRENT_PACKAGES_DIR}/include)

# Handle copyright
file(INSTALL "${SOURCE_PATH}/LICENSE" DESTINATION "${CURRENT_PACKAGES_DIR}/share/fftconv" RENAME copyright)