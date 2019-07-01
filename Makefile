MATH ?=math/
include math/make/libraries

CXXFLAGS+=-O3 -march=native -Ibenchmark/include -std=c++1y -Imath/ -I$(BOOST) -I$(SUNDIALS)/include -I$(EIGEN) -I$(TBB)/include -DSTAN_THREADS
LDLIBS+=-lbenchmark
LDFLAGS+=-Lbenchmark/build/src -L$(TBB)/lib -Wl,-rpath,"$(TBB)/lib" -ltbb -ltbbmalloc -ltbbmalloc_proxy
CXX = clang++

update: 
	git submodule update --init --recursive

benchmark/build/src/libbenchmark.a: benchmark benchmark/googletest update
	mkdir -p benchmark/build && cd benchmark/build && cmake .. -DCMAKE_BUILD_TYPE=RELEASE && make

benchmark/googletest:
	cd benchmark && git clone https://github.com/google/googletest
