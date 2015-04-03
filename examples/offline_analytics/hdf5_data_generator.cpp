#include <iostream>
#include <hdf5.h>
#include <hdf5_hl.h>

// Create a NetCDF file comprising a variable "point".
#define FILENAME "data.h5"
#define VARNAME "point"

#define NUM_ELEMS	1024

using namespace std;

int main() {
  hid_t fid;  // File ID.
  hsize_t dims[1] = {NUM_ELEMS};  // The length for each dimension.

  // Generate data.
  double *points = new double[NUM_ELEMS]; // Variable data.
  for (size_t i = 0; i < NUM_ELEMS; ++i) {
    points[i] = i;
  }
  cout << "The generated file contains " << NUM_ELEMS << " points." << endl;

  // Create the file.
  fid = H5Fcreate(FILENAME, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

  // Write data.
  H5LTmake_dataset_double(fid, VARNAME, 1, dims, points);

  H5Fclose(fid);

  delete [] points;

  cout << "The file " << FILENAME << " has been created." << endl;
  cout << "To view this file, enter the command \"h5dump " << FILENAME << "\"." << endl;

  return 0;
}
