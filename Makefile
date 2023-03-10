BUILD_DIR := ./build
SRC_DIR := ./src
INCLUDE_DIR := ./include
OUT	= OpenGL-All
CC	 = clang
NVCC	 = nvcc
FLAGS	 = -Wall -g -c -x c++ -Ofast
EFLAGS	 := -I $(INCLUDE_DIR)
CUDA_FLAGS	 = -g -c -O3
LFLAGS	 = -lm build/glfw/src/libglfw3.a -lGL -lGLEW -lGLU

.PHONY: all
all: $(OUT)
	@echo "Made $(OUT)"

# #####################
# #    Auto Checks    #
# #####################

# check if we have nvcc
ifeq (, $(shell which nvcc))
$(error "MISSING NVIDA CUDA, No nvcc in $(PATH)")
endif

# check if there is clang that works with cuda
# ifneq (, $(shell which clang-14))
# NVCC = clang++
# endif

# check if there is clang
ifeq (, $(shell which clang))
CC = gcc
endif

# ###################
# #    Get files    #
# ###################

SOURCE	= $(shell find $(SRC_DIR) -name '*.cpp' -or -name '*.c')
CUDA_SOURCE	= $(shell find $(SRC_DIR) -name '*.cu')
HEADS	= $(shell find $(INCLUDE_DIR) -name '*.hpp' -or -name '*.h' -or -name '*.cuh')

# #################################
# #    Make GLM, SPDLOG, IMGUI    #
# #################################

# IMGUI
SOURCE	+= $(shell ls exernalLibraries/imgui/*.cpp)
SOURCE	+= exernalLibraries/imgui/backends/imgui_impl_glfw.cpp
SOURCE	+= exernalLibraries/imgui/backends/imgui_impl_opengl3.cpp
HEADS	+= $(shell ls exernalLibraries/imgui/*.h)
HEADS	+= exernalLibraries/imgui/backends/imgui_impl_glfw.h
HEADS	+= exernalLibraries/imgui/backends/imgui_impl_opengl3.h
IMGUI_INCLUDE1_DIR	= exernalLibraries/imgui
IMGUI_INCLUDE2_DIR	= exernalLibraries/imgui/backends
EFLAGS += -I $(IMGUI_INCLUDE1_DIR) -I $(IMGUI_INCLUDE2_DIR)

# SPDLOG
# exernalLibraries/spdlog/include
SPDLOG_INCLUDE_DIR = exernalLibraries/spdlog/include
EFLAGS += -I $(SPDLOG_INCLUDE_DIR)

# STB
# exernalLibraries/stb
STB_INCLUDE_DIR = exernalLibraries/stb
EFLAGS += -I $(STB_INCLUDE_DIR)

# psd
# exernalLibraries/portable-file-dialogs
PFD_INCLUDE_DIR = exernalLibraries/portable-file-dialogs
EFLAGS += -I $(PFD_INCLUDE_DIR)

# GLFW lib
$(BUILD_DIR)/glfw/src/libglfw3.a:
	mkdir -p $(BUILD_DIR)/glfw
	cmake -B build/glfw exernalLibraries/glfw
	cmake --build build/glfw


# ###################
# #    Get OBJS     #
# ###################
OBJS	:= $(SOURCE:%=$(BUILD_DIR)/%.o) $(CUDA_SOURCE:%=$(BUILD_DIR)/%.o)
OBJS	+= $(BUILD_DIR)/glfw/src/libglfw3.a

# ######################################
# #Get argumetns for runing the program#
# ######################################

# If the first argument is "run"...
ifeq (run,$(firstword $(MAKECMDGOALS)))
  # use the rest as arguments for "run"
  RUN_ARGS := $(wordlist 2,$(words $(MAKECMDGOALS)),$(MAKECMDGOALS))
  # ...and turn them into do-nothing targets
  $(eval $(RUN_ARGS):;@:)
endif

# If the first argument is "debug"...
ifeq (debug,$(firstword $(MAKECMDGOALS)))
  # use the rest as arguments for "run"
  RUN_ARGS := $(wordlist 2,$(words $(MAKECMDGOALS)),$(MAKECMDGOALS))
  # ...and turn them into do-nothing targets
  $(eval $(RUN_ARGS):;@:)
endif

# If the first argument is "valgrind"...
ifeq (valgrind,$(firstword $(MAKECMDGOALS)))
  # use the rest as arguments for "run"
  RUN_ARGS := $(wordlist 2,$(words $(MAKECMDGOALS)),$(MAKECMDGOALS))
  # ...and turn them into do-nothing targets
  $(eval $(RUN_ARGS):;@:)
endif

# If the first argument is "valgrind_extreme"...
ifeq (valgrind_extreme,$(firstword $(MAKECMDGOALS)))
  # use the rest as arguments for "run"
  RUN_ARGS := $(wordlist 2,$(words $(MAKECMDGOALS)),$(MAKECMDGOALS))
  # ...and turn them into do-nothing targets
  $(eval $(RUN_ARGS):;@:)
endif

# ################
# #    Targets   #
# ################

.PHONY: info
info:
	echo "SOURCE: $(SOURCE)"
	echo "CUDA_SOURCE: $(CUDA_SOURCE)"
	echo "OBJS: $(OBJS)"
	echo "HEADER: $(HEADS)"

# Build program
$(OUT): $(OBJS)
ifeq (clang++, $(NVCC))
	clang++ -g $(OBJS) -o $(OUT) $(LFLAGS) -L/usr/local/cuda/lib64/ -lcudart_static -ldl -lrt -pthread
else
	$(NVCC) -g $(OBJS) -o $(OUT) $(LFLAGS)
endif

$(BUILD_DIR)/%.c.o: %.c $(HEADS)
	mkdir -p $(dir $@)
	$(CC) $(FLAGS) $< -o $@ $(EFLAGS)

$(BUILD_DIR)/%.cpp.o: %.cpp $(HEADS)
	mkdir -p $(dir $@)
ifeq (clang++, $(NVCC))
	$(NVCC) -Wall $(CUDA_FLAGS) $< -o $@ $(EFLAGS)
else
	$(NVCC) $(CUDA_FLAGS) $< -o $@ $(EFLAGS) --compiler-options -Wall
endif

$(BUILD_DIR)/%.cu.o: %.cu $(HEADS)
	mkdir -p $(dir $@)
ifeq (clang++, $(NVCC))
	$(NVCC) -Wall $(CUDA_FLAGS) $< -o $@ $(EFLAGS)
else
	$(NVCC) $(CUDA_FLAGS) $< -o $@ $(EFLAGS) --compiler-options -Wall
endif


# clean house
.PHONY: clean
clean:
	rm -rf $(OBJS) $(OUT) $(BUILD_DIR) *.log

# run the program
.PHONY: run
run: $(OUT)
	./$(OUT) $(RUN_ARGS)

.PHONY: debug
debug: $(OUT)
	valgrind --log-file="valgrind.log" ./$(OUT) $(RUN_ARGS)

.PHONY: valgrind
valgrind: $(OUT)
	valgrind --log-file="valgrind.log" --leak-check=full ./$(OUT) $(RUN_ARGS)

.PHONY: valgrind_extreme
valgrind_extreme: $(OUT)
	valgrind --log-file="valgrind.log" --leak-check=full --show-leak-kinds=all --leak-resolution=high --track-origins=yes --vgdb=yes ./$(OUT) $(RUN_ARGS)
