# DSE Precompute — Makefile
#
# Setup:
#   cp /path/to/your/{analytical_models.cpp,analytical_models.h,trace_analysis.cpp,trace_analysis.h} .
#   cat dse_precompute_impl.cpp >> analytical_models.cpp
#   make

CXX      = g++
CXXFLAGS = -O2 -std=c++17 -Wall
TARGET   = dse_precompute

$(TARGET): dse_main.cpp analytical_models.cpp trace_analysis.cpp \
           analytical_models.h trace_analysis.h dse_precompute.h
	$(CXX) $(CXXFLAGS) -o $@ dse_main.cpp analytical_models.cpp trace_analysis.cpp

clean:
	rm -f $(TARGET)
