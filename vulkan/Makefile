OPT_PRINT_VULKAN_LAYERS:=1
OPT_PRINT_VULKAN_EXTENSIONS:=1

CXX_FLAGS:=-O0 -g3 -ggdb -std=c++23 -Wall -Wextra -I/home/holz/etc/git/vulkan_headers/build/install/include/
LINK_FLAGS:=-lvulkan -lxcb

ifeq ($(OPT_PRINT_VULKAN_LAYERS),1)
	CXX_FLAGS+= -DOPT_PRINT_VULKAN_LAYERS
endif

ifeq ($(OPT_PRINT_VULKAN_EXTENSIONS),1)
	CXX_FLAGS+= -DOPT_PRINT_VULKAN_EXTENSIONS
endif

all: vkexplore

%.o: %.cc Makefile
	$(CXX) $< -c -o $@ $(CXX_FLAGS)

vkexplore: vkexplore.o
	$(CXX) $^ -o $@ $(LINK_FLAGS)

clean:
	rm -f vkexplore.o vkexplore

.PHONY: clean

