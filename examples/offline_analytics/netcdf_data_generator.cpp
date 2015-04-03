#include <iostream>
#include <netcdf.h>

// Create a NetCDF file comprising a variable "point".
#define FILENAME "data.nc"
#define VARNAME "point"

#define NUM_ELEMS	1024

using namespace std;

int main() {
  int fid;  // File ID.
  int dimid[1] = {NUM_ELEMS};  // Dimension ID.
  int varid;  // Variable ID.
  double *points = new double[NUM_ELEMS]; // Variable data.

  // Define the file schema.
  nc_create(FILENAME, NC_64BIT_OFFSET, &fid);
  nc_def_dim(fid, "len", NUM_ELEMS, &dimid[0]);
  nc_def_var(fid, VARNAME, NC_DOUBLE, 1, dimid, &varid);
  nc_enddef(fid);
  cout << "The generated file contains " << NUM_ELEMS << " points." << endl;

  // Generate data.
  for (size_t i = 0; i < NUM_ELEMS; ++i) {
    points[i] = i;
  }

  // Write data.
  nc_put_var_double(fid, varid, points);

  nc_close(fid);

  delete [] points;

  cout << "The file " << FILENAME << " has been created." << endl;
  cout << "To view this file, enter the command \"ncdump " << FILENAME << "\"." << endl;

  return 0;
}
