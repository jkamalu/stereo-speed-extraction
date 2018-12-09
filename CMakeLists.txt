CMAKE_MINIMUM_REQUIRED(VERSION 3.1)
set (CMAKE_CXX_STANDARD 11)
FIND_PACKAGE(OpenCV REQUIRED)

PROJECT(Stereo_Speed_Extractor)

ADD_EXECUTABLE(speed_test speed_test.cpp speed_test.h speed.cpp speed.h image.cpp image.h )
TARGET_LINK_LIBRARIES(speed_test ${OpenCV_LIBS})
