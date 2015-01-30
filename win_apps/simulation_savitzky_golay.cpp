#include <mpi.h>

#include "savitzky_golay.h"
#include "scheduler.h"

#define NUM_THREADS 4 // The # of threads for analytics task.
#define STEP  1 // The size of unit chunk for each single read, which groups a bunch of elements for mapping and reducing. (E.g., for a relational table, STEP should equal the # of columns.) 
#define NUM_ELEMS 1024 // The total number of elements of the simulated data.

#define PRINT_COMBINATION_MAP 1
#define PRINT_OUTPUT 1

int main(int argc, char* argv[]) {
  // MPI initialization.
  int mpi_status = MPI_Init(&argc, &argv);
  if (mpi_status != MPI_SUCCESS) {
    printf("Failed to initialize MPI environment.\n");
    MPI_Abort(MPI_COMM_WORLD, mpi_status);
  }

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  // Only used for time statistics, not necessarily added to the simulation code.
  chrono::time_point<chrono::system_clock> clk_beg, clk_end;
  clk_beg = chrono::system_clock::now();

  const size_t total_len = NUM_ELEMS;
  float* in = new float[total_len];
  const size_t out_len = NUM_ELEMS;
  double* out = new double[out_len];

  // Run the given simulation.
  for (size_t i = 0; i < total_len; ++i) {
    in[i] = i + rank;
  }

  clk_end = chrono::system_clock::now();
  std::chrono::duration<double> sim_seconds = clk_end - clk_beg;
  if (rank == 0) {
    printf("Simulation time = %.2f secs.\n", sim_seconds.count());
    printf("Simulation data is ready...\n");
  }

  // Insert in-situ processing code.
  if (rank == 0)
    printf("Run in-situ processing...\n");
  SchedArgs args(NUM_THREADS, STEP);
  std::unique_ptr<Scheduler<float, double>> win_app(new SavitzkyGolay<float, double>(args));
	win_app->set_red_obj_size(sizeof(WinObj));   
  win_app->set_glb_combine(false);
  win_app->run2(in, total_len, out, out_len);
  if (rank == 0)
    printf("In-situ processing is done.\n");

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

  // Only used for time statistics, not necessarily added to the simulation code.
  clk_end = chrono::system_clock::now();
  std::chrono::duration<double> elapsed_seconds = clk_end - clk_beg;
  if (rank == 0)
    printf("Analytics time = %.2f secs.\n", elapsed_seconds.count() - sim_seconds.count());
  printf("Total processing time on node %d = %.2f secs.\n", rank, elapsed_seconds.count());

  delete [] in;
  delete [] out;

  MPI_Finalize();

  return 0;
}
