#include <memory>
#include <mpi.h>
#include <typeinfo>

#include "hdf5_partitioner.h"
#include "logistic_regression.h"
#include "partitioner.h"
#include "scheduler.h"

#define NUM_THREADS 4  // The # of threads for analytics task.
// For logistic regression application, STEP and NUM_COLS in logistic_regression.h must be equal.
#define STEP  NUM_COLS  // The size of unit chunk for each single read, which groups a bunch of elements for mapping and reducing. (E.g., for a relational table, STEP should equal the # of columns.)
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
  const size_t out_len = 1;  // The output is only a single weight vector.
  double** out = new double*[1];
  out[0] = new double[NUM_DIMS];

  // Set up the initial weights.
  double weights[NUM_DIMS];
  for (int i = 0; i < NUM_DIMS; ++i) {
    weights[i] = (double)rand() / RAND_MAX;
  }
  if (rank == 0) {
    printf("\nInitial weights:\n");
    printVector(weights);
  }

  SchedArgs args(NUM_THREADS, STEP, (void*)weights, NUM_ITERS);
  unique_ptr<Scheduler<double, double*>> lr(new LogisticRegression<double>(args)); 
  lr->set_red_obj_size(sizeof(GradientObj));
  lr->run((const double*)p->get_data(), p->get_len(), out, out_len);

  // Print out the combination map if required.
  if (PRINT_COMBINATION_MAP && rank == 0) {
    printf("\n");
    lr->dump_combination_map();
  }

  // Print out the final result on the master node if required.
  if (PRINT_OUTPUT && rank == 0) {
    printf("Final output on the master node:\n");
    for (int i = 0; i < NUM_DIMS; ++i) {
      printf("%.2f ", out[0][i]);
    }
    printf("\n");
  }

  delete [] out[0];
  delete [] out;

  MPI_Finalize();

  return 0;
}
