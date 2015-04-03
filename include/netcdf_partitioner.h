#ifndef _NETCDF_PARTITIONER_H_                                                           
#define _NETCDF_PARTITIONER_H_

#include <string>

#include "partitioner.h"

class NetCDFPartitioner : public Partitioner {
 public:
  using Partitioner::Partitioner;

  // Initialize the variable data type.
  void init_vartype() override;

  // Load an array slab.
  void load(const size_t start[], const size_t count[], int ndims) override;

 protected:
  // Check the data format.
  bool checkFormat() const override {
    return "nc" == filename_.substr(filename_.size() - 2, 2);
  }

  // Get the dimensionality.
  int get_ndims() const override;

  // Get the length of each dimension.
  void get_dimlens(size_t dimlens[]) const override;

  // Get the total length of the 1D array associcated with the viriable.
  size_t get_varlen() const override;
};

#endif  // _NETCDF_PARTITIONER_H_
