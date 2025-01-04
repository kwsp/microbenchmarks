vcpkg_from_github(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO kwsp/fftconv
    REF c380acdc04aa3ce3f2bcfdbc7af1974b86f2974b  
    SHA512 e2f564612737795823163c84473f22ce0fb943ce38a4210a628db0d61da917981c0c87c866163fa98ebcd85af35585f2ff207122786faab29ab32ffdf358b4d3
    HEAD_REF main
)

# Copy header only lib
file(INSTALL ${SOURCE_PATH}/include/fftconv DESTINATION ${CURRENT_PACKAGES_DIR}/include)

# Handle copyright
file(INSTALL "${SOURCE_PATH}/LICENSE" DESTINATION "${CURRENT_PACKAGES_DIR}/share/fftconv" RENAME copyright)
