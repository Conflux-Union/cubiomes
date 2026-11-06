# cubiomes (Conflux-Union fork) - GPU module build and parity harness.
#
# This is NOT the fork's library build; the tracked build system is
# CMakeLists.txt (untouched by this file). This Makefile only builds the GPU
# engine plus the CPU-vs-GPU parity harness: `make` builds ./gpu_test, `make
# gpu-test` runs the gate. The CPU side is compiled with the determinism
# flags the mod's consumers use (-ffp-contract=off -fno-fast-math); the GPU
# kernel is a bit-exact executor and both sides must evaluate identical
# floating-point expressions.

CC := cc
# -DCUBIOMES_ENABLE_GPU compiles the transparent dispatch hook into
# biomenoise.c (mapOverworldSurfaceBiome tries the GPU engine first).
# Without the define the GPU code is unreachable through the public API and
# consumers that do not build gpu.c at all are unaffected.
CFLAGS := -O2 -Wall -MMD -MP -fno-fast-math -ffp-contract=off \
	-DCUBIOMES_ENABLE_GPU
LDLIBS := -lm -ldl -lpthread

# Compile list of the mod's native library (native/README.md) + GPU engine.
LIBSRC := \
	gpu.c \
	opencl_host.c \
	biomenoise.c \
	biomes.c \
	generator.c \
	layers.c \
	noise.c \
	terrainnoise.c \
	finders.c \
	terrain_features.c \
	util.c \
	features/end_city.c \
	features/fortress.c \
	features/abandoned_camp.c \
	features/stronghold.c

LIBOBJ := $(LIBSRC:.c=.o)
LIBA := libcubiomes.a

all: gpu_test

$(LIBA): $(LIBOBJ)
	ar rcs $@ $^

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

gpu_test: gpu_test.c $(LIBA)
	$(CC) $(CFLAGS) gpu_test.c $(LIBA) -o $@ $(LDLIBS)

# The parity gate: 0 mismatches over the default matrix; `./gpu_test --full`
# runs the full brief matrix. Skips cleanly (exit 0) without a usable OpenCL
# fp64 device.
gpu-test: gpu_test
	./gpu_test

clean:
	rm -f $(LIBA) gpu_test $(LIBOBJ) $(LIBOBJ:.o=.d) gpu_test.d

.PHONY: all gpu-test clean

-include $(LIBOBJ:.o=.d) gpu_test.d
