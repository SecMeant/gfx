OPENCL_VERSION:=300
CC:=clang
CCL:=clang
CFLAGS:=-Wall -Wextra -O2 -DCL_TARGET_OPENCL_VERSION=$(OPENCL_VERSION)
CCLFLAGS:=-O3
LDFLAGS:=-lOpenCL
SPIRV_LINKER:=llvm-spirv

# Some old loaders don't support SPIR-V,
# so by deafult we compile online.
ifeq ($(COMPILE_ONLINE),)
	COMPILE_ONLINE := 1
endif

CFLAGS += -DCOMPILE_ONLINE=$(COMPILE_ONLINE)

all: square square.spv

%.o: %.c Makefile
	$(CC) $< -c -o $@ $(CFLAGS)

square: square.o Makefile
	$(CC) $< -o $@ $(LDFLAGS)

square.bc: square.cl Makefile
	$(CCL) -c -target spir64 -emit-llvm -o $@ $< $(CCLFLAGS)

square.spv: square.bc Makefile
	$(SPIRV_LINKER) $< -o $@

clean:
	rm -f square square.o square.bc

.PHONY: clean

