#include "netcdf_partitioner.h"

#include <cstdlib>
#include <iostream>
#include <netcdf.h>
#include <string>
#include <typeinfo>

using namespace std;

void NetCDFPartitioner::init_vartype() {
  // Open the file and get the file ID.
  int ncid;
  nc_open(filename_.c_str(), NC_NOWRITE, &ncid);

  // Get the variable ID.
  int varid;
  nc_inq_varid(ncid, varname_.c_str(), &varid);	

  // Get the variable type.
  nc_type vartype;
  nc_inq_vartype(ncid, varid, &vartype);

  // Close the file.
  nc_close(ncid);

  switch (vartype) {
    case NC_INT:
      type_ = typeid(int).name();
      break;
    case NC_FLOAT:
      type_ = typeid(float).name();
      break;
    case NC_DOUBLE:
      type_ = typeid(double).name();
      break;
    default:
      printf("Error: Currently, only int/float/double types are supported.\n");
      exit(-1);
  } 
}

void NetCDFPartitioner::load(const size_t start[], const size_t count[], int ndims) {
  // Open the file and get the file ID.
  int ncid;
  nc_open(filename_.c_str(), NC_NOWRITE, &ncid);

  // Get the variable ID.
  int varid;
  nc_inq_varid(ncid, varname_.c_str(), &varid);	

  // Get the variable type.
  nc_type vartype;
  nc_inq_vartype(ncid, varid, &vartype);

  // Calculate the total length of the array slab.
  len_ = 1;
  for (int i = 0; i < ndims; ++i) {
    len_ *= count[i];
  }  

  // Load the array slab.
  switch (vartype) {
    case NC_INT:
      type_ = typeid(int).name();
      data_ = malloc(len_ * sizeof(int));
      nc_get_vara_int(ncid, varid, start, count, (int*)data_);
      break;
    case NC_FLOAT:
      type_ = typeid(float).name();
      data_ = malloc(len_ * sizeof(float));
      nc_get_vara_float(ncid, varid, start, count, (float*)data_); 
      break;
    case NC_DOUBLE:
      type_ = typeid(double).name();
      data_ = malloc(len_ * sizeof(double));
      nc_get_vara_double(ncid, varid, start, count, (double*)data_); 
      break;
    default:
      printf("Error: Currently, only int/float/double types are supported.\n");
      exit(-1);
  }

  // Close the file.
  nc_close(ncid);
}

int NetCDFPartitioner::get_ndims() const {
  // Open the file and get the file ID.
  int ncid;
  nc_open(filename_.c_str(), NC_NOWRITE, &ncid);

  // Get the variable ID.
  int varid;
  nc_inq_varid(ncid, varname_.c_str(), &varid);	
 
  // Get the dimensionality.
  int ndims;
  nc_inq_varndims(ncid, varid, &ndims);

  // Close the file.
  nc_close(ncid);

  return ndims;  
}

void NetCDFPartitioner::get_dimlens(size_t dimlens[]) const {
  // Open the file and get the file ID.
  int ncid;
  nc_open(filename_.c_str(), NC_NOWRITE, &ncid);

  // Get the variable ID.
  int varid;
  nc_inq_varid(ncid, varname_.c_str(), &varid);	
 
  // Get the dimensionality.
  int ndims;
  nc_inq_varndims(ncid, varid, &ndims);

  // Get the dimension IDs.
  int dimids[ndims];  // The ID of each dimension.
  nc_inq_vardimid(ncid, varid, dimids);
  
  for (int i = 0; i < ndims; ++i) {
    nc_inq_dimlen(ncid, dimids[i], &dimlens[i]);
  }

  // Close the file.
  nc_close(ncid);
}

size_t NetCDFPartitioner::get_varlen() const {
  // Open the file and get the file ID.
  int ncid;
  nc_open(filename_.c_str(), NC_NOWRITE, &ncid);

  // Get the variable ID.
  int varid;
  nc_inq_varid(ncid, varname_.c_str(), &varid);	
 
  // Get the dimensionality.
  int ndims;
  nc_inq_varndims(ncid, varid, &ndims);

  // Get the dimension IDs.
  int dimids[ndims];  // The ID of each dimension.
  nc_inq_vardimid(ncid, varid, dimids);
  
  size_t dimlens[ndims];  // The dimension lengths.
  size_t varlen = 1;
  for (int i = 0; i < ndims; ++i) {
    nc_inq_dimlen(ncid, dimids[i], &dimlens[i]);
    varlen *= dimlens[i];
  }

  // Close the file.
  nc_close(ncid);

  return varlen;
}
