#ifndef _PARTITIONER_H_                                                           
#define _PARTITIONER_H_

#include <iostream>
#include <mpi.h>

using namespace std;

class Partitioner {
 public:
  Partitioner(const string& filename, const string& varname, size_t step) : filename_(filename), varname_(varname), step_(step) {
    MPI_Comm_size(MPI_COMM_WORLD, &num_nodes_);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
  }

  // Initialize the variable data type.
  virtual void init_vartype() = 0;

  // Load the data partition for the current node.
  void load_partition(); 

  // Check the variable data type.
  bool is_vartype(const string& type) {
    if (type_ == "")
      init_vartype();

    return type_ == type;
  }

  // Get the partition length.
  size_t get_len() const {
    return len_;
  }

  // Get the partitioned data.
  void* get_data() const {
    return data_;
  }

  virtual ~Partitioner() {
    if (data_ != nullptr) {
      free(data_);
    }
  }

 protected:
  // Check the data format.
  virtual bool checkFormat() const = 0;

  // Get the dimensionality.
  virtual int get_ndims() const = 0;

  // Get the length of each dimension.
  virtual void get_dimlens(size_t dimlens[]) const = 0;
 
  // Get the total length of the 1D array associcated with the viriable.
  virtual size_t get_varlen() const = 0;

  // Load an array slab.
  virtual void load(const size_t start[], const size_t count[], int ndims) = 0;

  string filename_;
  string varname_;
  size_t step_;
  int num_nodes_;
  int rank_;

  string type_ = "";
  size_t len_ = 0;
  void* data_ = nullptr;
};

#endif  // _PARTITIONER_H_
