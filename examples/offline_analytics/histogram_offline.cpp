#include <memory>
#include <mpi.h>
#include <typeinfo>

#include "histogram.h"
#include "netcdf_partitioner.h"
#include "partitioner.h"
#include "scheduler.h"

#define NUM_THREADS 2  // The # of threads for analytics task.
#define STEP  1  // The size of unit chunk for each single read, which groups a bunch of elements for mapping and reducing. (E.g., for a relational table, STEP should equal the # of columns.) 
#define NUM_ELEMS 1024  // The total number of elements of the simulated data.

#define PRINT_COMBINATION_MAP 1
#define PRINT_OUTPUT  1

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

  // Check if the data type matches the input data of Smart scheduler.
  assert(p->is_vartype(typeid(double).name()));

  // Since the number of buckets in the histogram is unknown, here we do not
  // define an output array.
  SchedArgs args(NUM_THREADS, STEP);
  unique_ptr<Scheduler<double, size_t>> h(new Histogram<double>(args));
  h->set_red_obj_size(sizeof(Hist));
  h->run((const double*)p->get_data(), p->get_len(), nullptr, 0);  // Note that here the output array is nullptr.

  // Print out the combination map if required.
  // The final output can be transformed from the (global) combination map.
  if (PRINT_COMBINATION_MAP && rank == 0) {
    printf("\n");
    h->dump_combination_map();
  }

  // Print out the final result on the master node if required.
  if (PRINT_OUTPUT && rank == 0) {
    printf("Final output on the master node:\n");
    const auto& map = h->get_combination_map();
    for (const auto& pair : map) {
      const Hist* hist = static_cast<const Hist*>(pair.second.get());
      printf("%lu ", hist->count);
    }
    printf("\n");
  }

  MPI_Finalize();

  return 0;
}
