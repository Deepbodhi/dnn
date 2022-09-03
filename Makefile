SRC_DIR = ./source
SRC = $(wildcard $(SRC_DIR)/*.cpp)
OBJS_DIR =  ./build
OBJS = $(patsubst %.cpp,$(OBJS_DIR)/%.o,$(notdir $(SRC)))
TARGET = $(notdir $(shell pwd))
CC = g++

CFLAGS = -w -O0 -g -std=c++0x -fpermissive

OPENCV_INCLUDE  = /usr/local/Cellar/opencv/4.5.3_2/include/opencv4
OPENCV_LIB_PATH = /usr/local/Cellar/opencv/4.5.3_2/lib
OPENCV_LIBS     = -lopencv_stitching -lopencv_video -lopencv_videostab -lopencv_photo -lopencv_flann -lopencv_ml -lopencv_features2d -lopencv_superres -lopencv_objdetect -lopencv_calib3d -lopencv_imgproc -lopencv_core -lopencv_highgui -lopencv_imgcodecs -lopencv_videoio

INCLUDE = -I$(OPENCV_INCLUDE)
LIBS = -framework OpenCL
LIBS+= -L$(OPENCV_LIB_PATH) $(OPENCV_LIBS)

all:$(TARGET)

$(TARGET):$(OBJS)
	@echo Linking $(TARGET)...
	@$(CC) -rdynamic -o $@ $^  $(LIBS)

$(OBJS_DIR)/%.o:$(SRC_DIR)/%.cpp
	@echo Compiling $<
	@$(CC) $(CFLAGS) $(INCLUDE) -c $< -o $@

.PHONY:clean print
clean:
	@rm run.sh $(TARGET) $(OBJS)

print:
	@echo $(SRC)
	@echo $(OBJS)
	@echo $(TARGET)