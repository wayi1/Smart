#include "partitioner.h"

#include <cstdlib>
#include <cstring>
#include <string>
#include <iostream>

using namespace std;

void Partitioner::load_partition() {
  // Check the data format.
  if (!checkFormat() && rank_ == 0) {
    printf("Error: Unknown data format for the file \"%s\".\n", filename_.c_str());
    exit(-1);
  }

  // Assume that the total length value will not overflow the range of size_t.
  size_t total_len = get_varlen();
  if (total_len % step_ != 0 && rank_ == 0) {
    printf("Error: The total length must be a multiple of unit chunk length.\n");
    exit(-1);
  }

  int ndims = get_ndims();
  size_t start[ndims];
  memset(start, 0, ndims * sizeof(size_t));
  size_t count[ndims];
  get_dimlens(count);

  // Partition the data in the highest dimension.
  if (ndims == 1) {
    size_t total_pts = total_len / step_;  // The total number of input data points.
    count[0] = total_pts / num_nodes_ * step_;
    start[0] = rank_ * count[0];
    if (total_pts % num_nodes_ != 0 && rank_ == num_nodes_ - 1) {
      count[0] += total_pts % num_nodes_ * step_;
    }
  } else {
    size_t base_len = total_len / count[0];
    if (base_len % step_ != 0 && rank_ == 0) {
      printf("Error: For a multi-dimensional data, the total length / the highest dimension length must be a multiple of unit chunk length.\n");
      exit(-1);
    }
    size_t total_len_in_highest_dim = count[0];  // The total length of the highest dimension.
    count[0] = total_len_in_highest_dim / num_nodes_;
    start[0] = rank_ * count[0];
    if (total_len_in_highest_dim % num_nodes_ != 0 && rank_ == num_nodes_ - 1) {
      count[0] += total_len_in_highest_dim % num_nodes_;
    }
  }

  load(start, count, ndims);
}
