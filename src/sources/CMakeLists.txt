SET(NAME Sources)
SET(SOURCES initialisation.cpp inlet.cpp outlet.cpp
setGhostCells.cpp setting.cpp timeStep.cpp slipWall.cpp noSlipWall.cpp
symmetry.cpp homogeneousNeumann.cpp setGrid.cpp)

ADD_LIBRARY(${NAME} ${SOURCES})
TARGET_LINK_LIBRARIES(${NAME} Fvm)
