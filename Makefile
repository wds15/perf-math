CXXFLAGS+=-Imath/lib/eigen_3.3.3/ -Ibenchmark/include -std=c++1y -Imath/ -Imath/lib/boost_1.66.0 -O3 -Imath/lib/sundials_4.1.0/include 
LDLIBS+=-lbenchmark
LDFLAGS+=-Lbenchmark/build/src
CXX ?= clang++ -std=c++11 -DSTAN_THREADS

update: 
	git submodule update --init --recursive

benchmark/build/src/libbenchmark.a: benchmark benchmark/googletest update
	mkdir -p benchmark/build && cd benchmark/build && cmake .. -DCMAKE_BUILD_TYPE=RELEASE && make

benchmark/googletest:
	cd benchmark && git clone https://github.com/google/googletest
