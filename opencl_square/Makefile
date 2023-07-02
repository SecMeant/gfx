OPENCL_VERSION:=300
CC:=clang
CCL:=clang
CFLAGS:=-Wall -Wextra -O2 -DCL_TARGET_OPENCL_VERSION=$(OPENCL_VERSION)
CCLFLAGS:=-O3
LDFLAGS:=-lOpenCL

all: bruteforce bruteforce.bc

%.o: %.c Makefile
	$(CC) $< -c -o $@ $(CFLAGS)

bruteforce: bruteforce.o Makefile
	$(CC) $< -o $@ $(LDFLAGS)

bruteforce.bc: bruteforce.cl Makefile
	$(CCL) -c -target spir64 -emit-llvm -o $@ $< $(CCLFLAGS)

clean:
	rm -f bruteforce bruteforce.o bruteforce.bc

.PHONY: clean

