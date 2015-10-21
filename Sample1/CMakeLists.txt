# This is the root ritk CMakeLists file.
CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

#-----------------------------------------------------------------------------

SET(SAMPLE_EXECUTABLE OpenCV_Sample)
FIND_PACKAGE( OpenCV REQUIRED )
ADD_EXECUTABLE( ${SAMPLE_EXECUTABLE} main.cxx lena.jpg )
TARGET_LINK_LIBRARIES( ${SAMPLE_EXECUTABLE} ${OpenCV_LIBS} )

FILE(COPY
    ${CMAKE_CURRENT_SOURCE_DIR}/lena.jpg
    DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/debug)
	
FILE(COPY
    ${CMAKE_CURRENT_SOURCE_DIR}/lena.jpg
    DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/release)
		
FILE(COPY
    ${CMAKE_CURRENT_SOURCE_DIR}/lena.jpg
    DESTINATION ${CMAKE_CURRENT_BINARY_DIR})