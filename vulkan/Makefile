OPT_PRINT_VULKAN_LAYERS:=1

CXX_FLAGS:=-std=c++23 -Wall -Wextra -I/home/holz/etc/git/vulkan_headers/build/install/include/
LINK_FLAGS:=-lvulkan

ifeq ($(OPT_PRINT_VULKAN_LAYERS),1)
	CXX_FLAGS+= -DOPT_PRINT_VULKAN_LAYERS
endif

all: vkexplore

%.o: %.cc Makefile
	$(CXX) $< -c -o $@ $(CXX_FLAGS)

vkexplore: vkexplore.o
	$(CXX) $^ -o $@ $(LINK_FLAGS)

clean:
	rm -f vkexplore.o

.PHONY: clean

