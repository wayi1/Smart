#include <memory>
#include <mpi.h>
#include <typeinfo>

#include "moving_avg.h"
#include "netcdf_partitioner.h"
#include "partitioner.h"
#include "scheduler.h"

#define NUM_THREADS 4  // The # of threads for analytics task.
#define STEP  1  // The size of unit chunk for each single read, which groups a bunch of elements for mapping and reducing. (E.g., for a relational table, STEP should equal the # of columns.) 
#define NUM_ELEMS 1024  // The total number of elements of the simulated data.

#define PRINT_COMBINATION_MAP 1
#define PRINT_OUTPUT 1

#define FILENAME  "data.nc"
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
  unique_ptr<Partitioner> p(new NetCDFPartitioner(FILENAME, VARNAME, STEP));
  p->load_partition();

  const size_t out_len = NUM_ELEMS;
  double* out = new double[out_len];

  SchedArgs args(NUM_THREADS, STEP);
  unique_ptr<Scheduler<double, double>> win_app(new MovingAverage<double, double>(args));
  win_app->set_red_obj_size(sizeof(WinObj));   
  win_app->set_glb_combine(false);
  win_app->run2((const double*)p->get_data(), p->get_len(), out, out_len);

  // Print out the combination map if required.
  if (PRINT_COMBINATION_MAP && rank == 0) {
    printf("\n");
    win_app->dump_combination_map();
  }

  // Print out the final result on the master node if required.
  if (PRINT_OUTPUT && rank == 0) {
    printf("Final output on the master node:\n");
    for (size_t i = 0; i < out_len; ++i) {
      printf("%.2f ", out[i]);
    }
    printf("\n");
  }

  delete [] out;

  MPI_Finalize();

  return 0;
}
