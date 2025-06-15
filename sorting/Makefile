CXX ?= clang++
override CXX_FLAGS += \
	-Wall \
	-Wextra \
	-O2 \
	-std=c++20 \

ifeq ("$(V)", "1")
	Q := 
else
	Q := @
endif

all: sortbench

%.o: %.cc Makefile
	@echo -e "  CXX\t" $<
	$(Q)$(CXX) $< -c -o $@ $(CXX_FLAGS)

sortbench: sortbench.o Makefile
	@echo -e "  LINK\t" $@
	$(Q)$(CXX) $< -o $@

clean:
	$(Q)rm -f sortbench sortbench.o

.PHONY: clean

