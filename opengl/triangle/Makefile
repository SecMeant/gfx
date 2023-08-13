CC ?= gcc
CXX ?= g++
CFLAGS_BASE := -Wall -Wextra -O2 -std=c++23
CXX_LINK_FLAGS := -lGL -lglfw -lGLEW -lmipc

CFLAGS += $(CFLAGS_BASE)

all: triangle Makefile

triangle: triangle.cc.o
	$(CXX) $^ -o $@ $(CXX_LINK_FLAGS)

%.cc.o: %.cc
	$(CXX) $< -c -o $@ $(CFLAGS)

clean:
	rm -f triangle.cc.o

