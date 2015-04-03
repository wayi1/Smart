#include <memory>
#include <mpi.h>
#include <omp.h>

#include "moving_avg.h"
#include "scheduler.h"

#define NUM_SIMULATION_THREADS 2  // The # of threads for simulation task.
#define NUM_ANALYTICS_THREADS 6  // The # of threads for analytics task.
#define NUM_RUNS  10  // The # of simulation runs.

#define STEP  1  // The size of unit chunk for each single read, which groups a bunch of elements for mapping and reducing. (E.g., for a relational table, STEP should equal the # of columns.) 
#define NUM_ELEMS 1024  // The total number of elements of the simulated data.

#define PRINT_COMBINATION_MAP 1
#define PRINT_OUTPUT 1

#define GRID_SIZE 1000

using namespace std;

// Run the given simulation.
void simulate(float* in, size_t total_len, int num_threads) {
#pragma omp parallel for num_threads(num_threads)
  for (size_t i = 0; i < total_len; ++i) {
    in[i] = i % GRID_SIZE;
  }
}

int main(int argc, char* argv[]) {
  // MPI initialization.
  int mpi_status = MPI_Init(&argc, &argv);  // This MPI environment initialization is not thread-safe.

  /* If no global synchronization is required, then no need to upgrade thread level.
     int provided;  // Provided thread level.
     int request = MPI_THREAD_MULTIPLE;  // The thread level must be MPI_THREAD_MULTIPLE.
     int mpi_status = MPI_Init_thread(&argc, &argv, request, &provided);
     */
  if (mpi_status != MPI_SUCCESS) {
    printf("Failed to initialize MPI environment.\n");
    MPI_Abort(MPI_COMM_WORLD, mpi_status);
  }

  int num_nodes, rank;
  MPI_Comm_size(MPI_COMM_WORLD, &num_nodes); 
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  /* If no global synchronization is required, then no need to upgrade thread level.
     if (request != provided && rank == 0) {
     printf("Failed to initialize MPI thread level at %d, the current level is %d.\n", request, provided);

     printf("Thread level information:\n");
     printf("MPI_THREAD_SINGLE = %d\n",  MPI_THREAD_SINGLE);
     printf("MPI_THREAD_FUNNELED = %d\n",  MPI_THREAD_FUNNELED);
     printf("MPI_THREAD_SERIALIZED = %d\n",  MPI_THREAD_SERIALIZED);
     printf("MPI_THREAD_MULTIPLE = %d\n",  MPI_THREAD_MULTIPLE);

     MPI_Abort(MPI_COMM_WORLD, mpi_status);
     }
     */

  size_t total_len = NUM_ELEMS;
  float* in = new float[total_len];
  size_t out_len = NUM_ELEMS;
  double* out = new double[out_len];

  omp_set_nested(1);  // Make sure nested parallism is on. 
  int num_procs = omp_get_num_procs();
  int num_threads1 = NUM_SIMULATION_THREADS;
  int num_threads2 = NUM_ANALYTICS_THREADS;
  if (rank == 0)
    printf("# of procs = %d, # of threads on task1 = %d, # of threads on task 2 = %d.\n", num_procs, num_threads1, num_threads2);
  int num_tasks = 2;
  int num_runs = NUM_RUNS;
  if (rank == 0)
    printf("# of simulation runs = %d.\n", num_runs);

  SchedArgs args(num_threads2, STEP); 
  unique_ptr<Scheduler<float, double>> win_app(new MovingAverage<float, double>(args));
  win_app->set_red_obj_size(sizeof(WinObj));
  win_app->set_glb_combine(false);

#pragma omp parallel num_threads(num_tasks)
#pragma omp single
  {
#pragma omp task  // Simulation task.
    {
      for (int i = 0; i < num_runs; ++i) {
        // Run simulation in parallel.
        simulate(in, total_len, num_threads1);

        // Some extra code involving MPI.
        int tag = 0;
        if (rank == 0) {
          MPI_Send(&num_nodes, 1, MPI_INT, 1, tag, MPI_COMM_WORLD);
        } else if (rank == 1) {
          MPI_Status status;
          int temp;
          MPI_Recv(&temp, 1, MPI_INT, 0, tag, MPI_COMM_WORLD, &status);
        }

        // Only a single thread runs the feeding process.
        win_app->feed(in, total_len);

        if (rank == 0)
          printf("Simulation data in run%d is ready...\n", i);
      }
    }

#pragma omp task  // Analytics task.
    {
      for (int i = 0; i < num_runs; ++i) {
        win_app->run2(out, out_len);

        if (rank == 0)
          printf("In-situ processing for run%d is done.\n", i);

        // Print out the combination map if required.
        // The final output can be transformed from the (global) combination map.
        if (PRINT_COMBINATION_MAP && rank == 0) {
          printf("\n");
          win_app->dump_combination_map();
        }

        // Print out the final result on the master node if required.
        if (PRINT_OUTPUT && rank == 0) {
          printf("Final output on the master node:\n");
          for (size_t i = 0; i < out_len; ++i) {
            printf("%.2f ", out[0]);
          }
          printf("\n");
        }
      }
    }
  } 

  delete [] in;
  delete [] out;

  MPI_Finalize();

  return 0;
}
