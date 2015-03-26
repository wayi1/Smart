#include "hdf5_partitioner.h"

#include <cstdlib>
#include <iostream>
#include <hdf5.h>
#include <string>
#include <typeinfo>

using namespace std;

void HDF5Partitioner::init_vartype() {
  // Open the file and get the file ID.
  hid_t fid = H5Fopen(filename_.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);

  // Get the variable ID.
  hid_t varid = H5Dopen(fid, varname_.c_str(), H5P_DEFAULT);

  // Get the variable type.
  hid_t vartype = H5Dget_type(varid);
  H5T_class_t t_class = H5Tget_class(vartype);

  if (t_class == H5T_INTEGER) {
    type_ = typeid(int).name();
  } else if (t_class == H5T_FLOAT && H5Tget_size(vartype) == 4) {
      type_ = typeid(float).name();
  } else if (t_class == H5T_FLOAT && H5Tget_size(vartype) == 8) {
    type_ = typeid(double).name();
  } else {
    printf("Error: Currently, only int/float/double types are supported.\n");
    exit(-1);
  }

  // Close handlers.
  H5Dclose(varid);
  H5Tclose(vartype);
  H5Fclose(fid);
}

void HDF5Partitioner::load(const size_t start[], const size_t count[], int ndims) {
  // Open the file and get the file ID.
  hid_t fid = H5Fopen(filename_.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);

  // Get the variable ID.
  hid_t varid = H5Dopen(fid, varname_.c_str(), H5P_DEFAULT);

  // Select the array slab to load.
  hsize_t hstart[ndims];
  hsize_t hcount[ndims];
  hsize_t total_1d[1] = {1};
  for (int i = 0; i < ndims; ++i) {
    hstart[i] = start[i];
    hcount[i] = count[i];
    total_1d[0] *= count[i];
  }
  len_ = total_1d[0];

  // Select the array slab to load.
  hid_t dataspace = H5Dget_space(varid);
  H5Sselect_hyperslab(dataspace, H5S_SELECT_SET, hstart, NULL, hcount, NULL);

  // Create the memory space.
  hid_t memspace = H5Screate_simple(1, total_1d, NULL);

  // Get the variable type.
  hid_t vartype = H5Dget_type(varid);
  H5T_class_t t_class = H5Tget_class(vartype);

  if (t_class == H5T_INTEGER) {
    type_ = typeid(int).name();
    data_ = malloc(len_ * sizeof(int));
    H5Dread(varid, H5T_NATIVE_INT, memspace, dataspace, H5P_DEFAULT, (int*)data_);
  } else if (t_class == H5T_FLOAT && H5Tget_size(vartype) == 4) {
    type_ = typeid(float).name();
    data_ = malloc(len_ * sizeof(float));
    H5Dread(varid, H5T_NATIVE_FLOAT, memspace, dataspace, H5P_DEFAULT, (float*)data_);
  } else if (t_class == H5T_FLOAT && H5Tget_size(vartype) == 8) {
    type_ = typeid(double).name();
    data_ = malloc(len_ * sizeof(double));
    H5Dread(varid, H5T_NATIVE_DOUBLE, memspace, dataspace, H5P_DEFAULT, (double*)data_); 
  } else {
    printf("Error: Currently, only int/float/double types are supported.\n");
    exit(-1);
  }

  // Close handlers.
  H5Dclose(varid);
  H5Sclose(dataspace);
  H5Sclose(memspace);
  H5Tclose(vartype);
  H5Fclose(fid);
}

int HDF5Partitioner::get_ndims() const {
  // Open the file and get the file ID.
  hid_t fid = H5Fopen(filename_.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);

  // Get the variable ID.
  hid_t varid = H5Dopen(fid, varname_.c_str(), H5P_DEFAULT);

  // Get the dimensionality.
  hid_t dataspace = H5Dget_space(varid);
  int ndims = H5Sget_simple_extent_ndims(dataspace);

  // Close handlers.
  H5Dclose(varid);
  H5Sclose(dataspace);
  H5Fclose(fid);

  return ndims;  
}

void HDF5Partitioner::get_dimlens(size_t dimlens[]) const {
  // Open the file and get the file ID.
  hid_t fid = H5Fopen(filename_.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);

  // Get the variable ID.
  hid_t varid = H5Dopen(fid, varname_.c_str(), H5P_DEFAULT);

  // Get the dimensionality.
  hid_t dataspace = H5Dget_space(varid);
  int ndims = H5Sget_simple_extent_ndims(dataspace);

  hsize_t dims[ndims];
  H5Sget_simple_extent_dims(dataspace, dims, NULL);
  for (int i = 0; i < ndims; ++i) {
    dimlens[i] = dims[i];
  }

  // Close handlers.
  H5Dclose(varid);
  H5Sclose(dataspace);
  H5Fclose(fid);
}

size_t HDF5Partitioner::get_varlen() const {
  // Open the file and get the file ID.
  hid_t fid = H5Fopen(filename_.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);

  // Get the variable ID.
  hid_t varid = H5Dopen(fid, varname_.c_str(), H5P_DEFAULT);

  // Get the dimensionality.
  hid_t dataspace = H5Dget_space(varid);
  int ndims = H5Sget_simple_extent_ndims(dataspace);

  hsize_t dims[ndims];
  size_t varlen = 1;
  H5Sget_simple_extent_dims(dataspace, dims, NULL);
  for (int i = 0; i < ndims; ++i) {
    varlen *= dims[i];
  }

  // Close handlers.
  H5Dclose(varid);
  H5Sclose(dataspace);
  H5Fclose(fid);

  return varlen;
}
