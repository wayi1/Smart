#------------------------------------------------------------------------------
# Copyright (c) 2015, The Ohio State University
# All rights reserved.
# Author: Yi Wang
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#     * Neither the name of The Ohio University nor the names of its 
#       contributors may be used to endorse or promote products derived from 
#       this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE OHIO STATE UNIVERSITY ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE OHIO STATE UNIVERSITY BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#------------------------------------------------------------------------------ 


MPI_CC = mpicxx 
CFLAGS = -Wall -O3 -std=c++11 -static-libstdc++
OPENMP_LIB = -fopenmp

HOME = ../..
COMMON_HEADER_DIR = examples/common_app_headers
INC_DIR = include

PROGS = histogram_space_sharing kmeans_space_sharing logistic_regression_space_sharing moving_avg_space_sharing moving_median_space_sharing savitzky_golay_space_sharing 

all: $(PROGS)

histogram_space_sharing: histogram_space_sharing.cpp
	$(MPI_CC) $(CFLAGS) $^ -o $@ $(OPENMP_LIB) -I$(HOME)/$(COMMON_HEADER_DIR) -I$(HOME)/$(INC_DIR) 

kmeans_space_sharing: kmeans_space_sharing.cpp
	$(MPI_CC) $(CFLAGS) $^ -o $@ $(OPENMP_LIB) -I$(HOME)/$(COMMON_HEADER_DIR) -I$(HOME)/$(INC_DIR) 

logistic_regression_space_sharing: logistic_regression_space_sharing.cpp
	$(MPI_CC) $(CFLAGS) $^ -o $@ $(OPENMP_LIB) -I$(HOME)/$(COMMON_HEADER_DIR) -I$(HOME)/$(INC_DIR) 

moving_avg_space_sharing: moving_avg_space_sharing.cpp
	$(MPI_CC) $(CFLAGS) $^ -o $@ $(OPENMP_LIB) -I$(HOME)/$(COMMON_HEADER_DIR) -I$(HOME)/$(INC_DIR) 

moving_median_space_sharing: moving_median_space_sharing.cpp
	$(MPI_CC) $(CFLAGS) $^ -o $@ $(OPENMP_LIB) -I$(HOME)/$(COMMON_HEADER_DIR) -I$(HOME)/$(INC_DIR) 

savitzky_golay_space_sharing: savitzky_golay_space_sharing.cpp
	$(MPI_CC) $(CFLAGS) $^ -o $@ $(OPENMP_LIB) -I$(HOME)/$(COMMON_HEADER_DIR) -I$(HOME)/$(INC_DIR) 

clean:
	rm  -f *.o $(PROGS)
