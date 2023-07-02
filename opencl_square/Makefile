OPENCL_VERSION:=300
CC:=clang
CCL:=clang
CFLAGS:=-Wall -Wextra -O2 -DCL_TARGET_OPENCL_VERSION=$(OPENCL_VERSION)
CCLFLAGS:=-O3
LDFLAGS:=-lOpenCL

all: square square.bc

%.o: %.c Makefile
	$(CC) $< -c -o $@ $(CFLAGS)

square: square.o Makefile
	$(CC) $< -o $@ $(LDFLAGS)

square.bc: square.cl Makefile
	$(CCL) -c -target spir64 -emit-llvm -o $@ $< $(CCLFLAGS)

clean:
	rm -f square square.o square.bc

.PHONY: clean

