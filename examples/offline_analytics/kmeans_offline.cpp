#include <memory>
#include <mpi.h>
#include <typeinfo>

#include "hdf5_partitioner.h"
#include "kmeans.h"
#include "partitioner.h"
#include "scheduler.h"

#define NUM_THREADS 4  // The # of threads for analytics task.
// For k-means application, STEP and NUM_DIMS in kmeans.h must be equal. 
#define STEP  NUM_DIMS  // The size of unit chunk for each single read, which groups a bunch of elements for mapping and reducing. (E.g., for a relational table, STEP should equal the # of columns.) 
#define NUM_ELEMS 1024  // The total number of elements of the simulated data.
#define NUM_ITERS 2  // The # of iterations.

#define PRINT_COMBINATION_MAP 1
#define PRINT_OUTPUT 1

#define FILENAME  "data.h5"
#define VARNAME "point"

using namespace std;

int main(int argc, char* argv[]) {
  // MPI initialization.
  int mpi_status = MPI_Init(&argc, &argv);
  if (mpi_status != MPI_SUCCESS) {
    printf("Failed to initialize MPI environment.\n");
    MPI_Abort(MPI_COMM_WORLD, mpi_status);
  }

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  // Load the data partition.
  unique_ptr<Partitioner> p(new HDF5Partitioner(FILENAME, VARNAME, STEP));
  p->load_partition();

  // Check if the data type matches the input data of Smart scheduler.
  assert(p->is_vartype(typeid(double).name()));

  // The output is a 2D array that indicates k vectors in a multi-dimensional
  // space.
  const size_t out_len = NUM_MEANS;
  double** out = new double*[out_len];
  for (size_t i = 0; i < out_len; ++i) {
    out[i] = new double[NUM_DIMS];
  }

  // Set up the initial k centroids.
  double** means = new double*[NUM_MEANS];
  for (int i = 0; i < NUM_MEANS; ++i) {
    means[i] = new double[NUM_DIMS];
  }
  for (int i = 0; i < NUM_MEANS; ++i) {
    for (int j = 0; j < NUM_DIMS; ++j) {
      means[i][j] = i * 10;  // This setting can result in some empty clusters.
    }
  }

  SchedArgs args(NUM_THREADS, STEP, (void*)means, NUM_ITERS);
  unique_ptr<Scheduler<double, double*>> kmeans(new Kmeans<double>(args));   
  kmeans->set_red_obj_size(sizeof(ClusterObj<double>));
  kmeans->run((const double*)p->get_data(), p->get_len(), out, out_len);

  // Print out the combination map if required.
  if (PRINT_COMBINATION_MAP && rank == 0) {
    printf("\n");
    kmeans->dump_combination_map();
  }

  // Print out the final result on the master node if required.
  if (PRINT_OUTPUT && rank == 0) {
    printf("Final output on the master node:\n");
    for (size_t i = 0; i < out_len; ++i) {
      for (int j = 0; j < NUM_DIMS; ++j) {
        printf("%.2f ", out[i][j]);
      }
      printf("\n");
    }
    printf("\n");
  }

  for (size_t i = 0; i < out_len; ++i) {
    delete [] out[i];
  }
  delete [] out;

  for (int i = 0; i < NUM_MEANS; ++i) {
    delete [] means[i];
  }
  delete [] means;

  MPI_Finalize();

  return 0;
}
