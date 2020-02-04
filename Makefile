MATH ?=math/
include $(MATH)make/standalone

STAN ?=stan/

CXXFLAGS+=-O3 -march=native -Ibenchmark/include -I. -I$(STAN)/src
LDLIBS+=-lbenchmark
LDFLAGS+=-Lbenchmark/build/src
CXX ?= clang++

update: 
	git submodule update --init --recursive

benchmark/build/src/libbenchmark.a: benchmark benchmark/googletest update
	mkdir -p benchmark/build && cd benchmark/build && cmake .. -DCMAKE_BUILD_TYPE=RELEASE && make

benchmark/googletest:
	cd benchmark && git clone https://github.com/google/googletest
