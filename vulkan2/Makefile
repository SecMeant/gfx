GLSLC:=glslc
CXX:=g++
CC:=gcc
CXX_FLAGS:=-O0 -g3 -ggdb -std=c++23 -Wall -Wextra -I/home/holz/etc/git/vulkan_headers/build/install/include/
LINK_FLAGS:=-lglfw -lvulkan -ldl -lpthread -lX11 -lXxf86vm -lXrandr -lXi

all: vkexplore shaders

shaders: shaders/ shaders/frag.spv shaders/vert.spv

shaders/:
	mkdir -p shaders/

shaders/vert.spv: shader.vert Makefile
	$(GLSLC) $< -o $@

shaders/frag.spv: shader.frag Makefile
	$(GLSLC) $< -o $@

%.o: %.cc Makefile
	$(CXX) $< -c -o $@ $(CXX_FLAGS)

vkexplore: vkexplore.o
	$(CXX) $^ -o $@ $(LINK_FLAGS)

clean:
	rm -f vkexplore.o vkexplore shaders/frag.spv shaders/vert.spv
	rmdir shaders

.PHONY: clean shaders

