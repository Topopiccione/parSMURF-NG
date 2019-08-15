# parSMURFng
# 2019 - Alessandro Petrini - AnacletoLAB - Universita' degli Studi di Milano
project(parSMURFng)
cmake_minimum_required(VERSION 3.0.2)

# This time, MPI is a requirement of the project
find_package(MPI)
include_directories(${MPI_INCLUDE_PATH})

add_subdirectory(ann_1.1.2)
add_subdirectory(jsoncons)
add_subdirectory(zlib)
add_subdirectory(easyloggingpp)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")
include_directories(ann_1.1.2/include/ANN)
include_directories(ranger)
include_directories(.)
include_directories(ranger/Utility)
include_directories(ranger/Forest)
include_directories(ranger/Tree)
include_directories(${jsoncons_SOURCE_DIR})
include_directories(${zlib_SOURCE_DIR})
include_directories(easyloggingpp)

SET(SRCCOMMON				"ArgHandler_new.cpp"
							#"fileImport.cpp"
							"curves.cpp"
							"Folds.cpp"
							#"testtraindivider.cpp"
							"parSMURFUtils.cpp"
							"organizer.cpp"
							#"partition.cpp"
							"sampler.cpp"
							"rfRanger.cpp"
							"MegaCache.cpp"
							"hyperSMURF_core.cpp")

FILE(GLOB_RECURSE RANGER	"ranger/*.cpp")

SET(LOGGERSRC 				"easyloggingpp/easylogging++.cc")

SET(SOURCESMAIN				"parSMURFng.cpp"
							#"samplerKNN.cpp"
						#"hyperSMURF.cpp"
					)


if(NOT CMAKE_BUILD_TYPE)
	set(CMAKE_BUILD_TYPE Release)
endif()

ADD_CUSTOM_TARGET(debug
	COMMAND ${CMAKE_COMMAND} -DCMAKE_BUILD_TYPE=Debug ${CMAKE_SOURCE_DIR}
	COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target all
	COMMENT "Switch CMAKE_BUILD_TYPE to Debug"
)
ADD_CUSTOM_TARGET(release
	COMMAND ${CMAKE_COMMAND} -DCMAKE_BUILD_TYPE=Release ${CMAKE_SOURCE_DIR}
	COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target all
	COMMENT "Switch CMAKE_BUILD_TYPE to Release"
)

add_executable(parSMURFng ${SRCCOMMON} ${RANGER} ${SOURCESMAIN} ${LOGGERSRC})
add_dependencies(parSMURFng ANN)
add_dependencies(parSMURFng jsoncons)
add_dependencies(parSMURFng zlib)
add_dependencies(parSMURFng easyloggingpp)

## Per MPI
set_target_properties(parSMURFng PROPERTIES COMPILE_FLAGS "${MPI_COMPILE_FLAGS}")
set_target_properties(parSMURFng PROPERTIES LINK_FLAGS "${MPI_LINK_FLAGS}")
target_link_libraries(parSMURFng ANN ${zlib_LIB} ${MPI_LIBRARIES})
