#ifndef _SCHEDULER_H_                                                           
#define _SCHEDULER_H_

#include <cassert>
//#include <chrono>  // Only used if some timing code (for the purpose of profiling) is uncommented.
#include <cstring>
#include <ctime>
#include <map>
#include <memory>
#include <mpi.h>
#include <omp.h>
#include <sstream>
#include <unistd.h>
#include <vector>

#include "chunk.h"

using namespace std;

/* Debug printf */
#define dprintf(...)  //printf(__VA_ARGS__)

// Base reduction object.
struct RedObj {
  // Used for applications invovling multiple iterations.
  virtual void reset() {}

  // May also need to implement the copy constructor
  // if any reference-type member exists.
  virtual RedObj* clone() {
		return new RedObj(*this);
	}

  // Convert itself to a string.
  // This function is called in Scheduler's dump function.
  virtual string str() const {
    stringstream ss;
    ss << this;  
    return ss.str();
  }

  // Trigger to emit reduction object.
  // The default setting does not trigger anything, so it always returns false.
  virtual bool trigger() const {
    return false;
  }

  virtual ~RedObj() {}
};

// Scheduler arguments.
struct SchedArgs {
  int num_threads;
  size_t step;
  const void* extra_data;
  int num_iters;

  SchedArgs(int n, size_t s, const void* e = nullptr, int n_iters = 1) : num_threads(n), step(s), extra_data(e), num_iters(n_iters) {}
};

// Mapping mode, including 1-1 mapping and 1-n mapping (flatmap).
enum MAPPING_MODE_T {
  GEN_ONE_KEY_PER_CHUNK,
  GEN_KEYS_PER_CHUNK,
};

template <class In, class Out>
class Scheduler {
 public:
  /*
   * Public APIs.
   */
  explicit Scheduler(const SchedArgs& args);

  /* Run Functions for Space Sharing Mode */
  // Run the data processing by specifying both buffer size and mapping mode.
  // The input array's total length is the length of the data shared buffer used
  // for in-situ processing.
  // The buf_size can be set at most equal to total_len, which results in 
  // processing only a single input data chunk.
  // The mode specifies the mapping mode, i.e., generating a single key or
  // multiple keys per data chunk. 
  void run_space_sharing(const In* data, size_t total_len, size_t buf_size, Out* out, size_t out_len, MAPPING_MODE_T mode);

  // Run the data processing by generating a single key per data chunk.
  // buf_size_ equals the input array's total length by default.
  // This run function is preferred when no read buffer size is specified. 
  void run(const In* data, size_t total_len, Out* out, size_t out_len) {
	  run_space_sharing(data, total_len, total_len, out, out_len, GEN_ONE_KEY_PER_CHUNK);
  }

  // Run the data processing by generating multiple keys per data chunk.
  // buf_size_ equals the input array's total length by default.
  // This run function is preferred when no read buffer size is specified. 
  void run2(const In* data, size_t total_len, Out* out, size_t out_len) {
	  run_space_sharing(data, total_len, total_len, out, out_len, GEN_KEYS_PER_CHUNK);
  }

  /* Run Functions for Time Sharing Mode */
  // Run the data processing by specifying mapping mode.
  // The input array MUST be fed to circular_buf_ by a simulation process in advance.
  // buf_size_ equals the input array's total length by default.
  // The mode specifies the mapping mode, i.e., generating a single key or
  // multiple keys per data chunk. 
  void run_time_sharing(Out* out, size_t out_len, MAPPING_MODE_T mapping_mode);

  // Run the data processing by generating a single key per data chunk.
  // buf_size_ equals the input array's total length by default.
  void run(Out* out, size_t out_len) {
    run_time_sharing(out, out_len, GEN_ONE_KEY_PER_CHUNK);   
  }

  // Run the data processing by generating multiple keys per data chunk.
  // buf_size_ equals the input array's total length by default.
  void run2(Out* out, size_t out_len) {
    run_time_sharing(out, out_len, GEN_KEYS_PER_CHUNK);
  }
 
  // Feed a time step to circular_buf_.
  // This function MUST be called by a single thread. 
  // This function is used in time sharing mode only.
  void feed(const In* data, size_t total_len);

  /* Getters */
  // This function is used for converting the global combination map (i.e., global
  // results) into the desired output.
  // The global combination map is the combination map on the master node.
  // Thus, this function should only be called by the master node outside.
  const map<int, unique_ptr<RedObj>>& get_combination_map() const {
    return combination_map_;
  }

  // Retrieve num_iters_.
  int get_num_iters() const {
    return num_iters_;
  }

  // Retrieve num_threads_.
  int get_num_threads() const {
    return num_threads_;
  }

  // Retrieve glb_combine_.
  bool get_glb_combine() const {
    return glb_combine_;
  }

  /* Setters */
  // Set the derived reduciton object size.
  void set_red_obj_size(size_t size) {
    red_obj_size_ = size;
  }

  // Set glb_combine_.
  void set_glb_combine(bool flag) {
    glb_combine_ = flag;
  }

  /* Debugging Functions */
  void dump_reduction_map() const;
  void dump_combination_map() const;

  /* 
   * Customization Required for the Following Virtual Functions.
   */
  // A derivation on either gen_key or gen_keys is required.
  // Generate an (integer) key given a chunk.
  virtual int gen_key(const Chunk& chunk) const {return -1;}

  // Generate (integer) keys given a chunk.
  virtual void gen_keys(const Chunk& chunk, vector<int>& keys) const {}

  // Accumulate the chunk on a reduction object.
  virtual void accumulate(const Chunk& chunk, unique_ptr<RedObj>& red_obj) = 0;

  // Merge the first reduction object into the second reduction object, i.e.,
  // combination object.
	virtual void merge(const RedObj& red_obj, unique_ptr<RedObj>& com_obj) = 0;

  // Process extra data to help initialize combination_map_.
  virtual void process_extra_data() {}

  // Perform post-combination processing.
  // This function will only be applied to the global combination map.
  // Thus, this function will only be called by the master node.
  virtual void post_combine() {}

  // Construct a reduction object from serialized reduction object.
  // Usually only a trivial implementation is needed.
  virtual void deserialize(unique_ptr<RedObj>& obj, const char* data) const = 0;

  // Convert a reduction object to an output result.
  virtual void convert(const RedObj& red_obj, Out* out) const {} 

  /*
   * Internal APIs for Hacking.
   */
  // Set up data processing by binding the input data.
	void setup(const In* data, size_t total_len, size_t buf_size, Out* out, size_t out_len);
 
  // Process an input data chunk in the buffer, by multi-threading.
  // The input data chunk size is the total size before splitting.
  // If the buffer is full, which occurs almost all the time,
  // then the chunk's length equals the buffer.
  // Otherwise the buffer is not full, which only occurs
  // when the entire data cannot be divided by the buffer size and the process
  // is for the chunk.
  // In this case, the chunk's length will be less than the buffer.
  void process_chunk(const Chunk& input, MAPPING_MODE_T mode);

  // Copy global combination map to each (local) combination_map_.
  // Since global combiantion map is combination_map_ on the master node,
  // the master node will do nothing here.
  void distribute_global_combination_map();

  // Copy combination_map_ to each local reduction map.
  void distribute_local_combination_map();

  // Combine all the local reduction maps into a combination map.
  // This function should only be called by the master thread on each node.
  void local_combine();

  // Combine all the combination maps on slave nodes
  // into a global combination map on the master node.
  // This function can be further optimized by multithreading.
  void global_combine();

  // Output (local) combination_map_ to out_ on each node.
  // Usually the key in the map serves as the array index of out.
  // To output a reduction object as an output element value, certain
  // lightweigth conversion can be involved.
  virtual void output();

  virtual ~Scheduler() {
    // Clean the circular buffer if needed.
    for (size_t i = 0; i < CIRCULAR_BUF_SIZE_ && circular_buf_[i] != nullptr; ++i) {
      delete [] circular_buf_[i];
    }  
  }

 protected:
  // Calculate the start location for each data split.
  // Even partitioning is used by default.
  // This function should only be called by the master thread.
  virtual void split();

  // Retrieve the next unit chunk in the local split.
  // Return true if the last unit chunk in the local split has been retrieved.
  // The next unit chunk is retrieve based on step_ in default, without skipping any element.
  virtual bool next(unique_ptr<Chunk>& unit_chunk, const Chunk& split) const;

  // Perform the local reduction.
	void reduce(MAPPING_MODE_T mode);

  int num_threads_;
  int num_nodes_;
  int rank_;

  /* Input Array Data */
  const In* data_ = nullptr;
  size_t total_len_ = 0;
  Chunk input_; // Input layout on the data source provided by data_.
  size_t buf_size_ = 0;  // Input data buffer, which will be greater or equal to the length of input_ (not full or full).

  /* Output Array Data */
  Out* out_ = nullptr;
  size_t out_len_ = 0; 

  /* Scheduler Shared Data */
  size_t step_;  // The size of unit chunk for each single read, which groups a bunch of elements for mapping and reducing.
  const void* extra_data_;  // Extra input data, e.g., initial k centrioids in k-means application.
  int num_iters_;  // The # of iterations.
  size_t red_obj_size_;  // The size of the derived reduction object, used for reduction object serialization and message passing.
  // This field should be removed once the reduction object type is replaced by protocol buffer.
  vector<Chunk> splits_;  // The length equals num_threads_.
  vector<map<int, unique_ptr<RedObj>>> reduction_map_;  // The vector length equals the number of threads, and each map is a local reduction map.
  map<int, unique_ptr<RedObj>> combination_map_;  // (Local) combination map which holds the local combination results, after global combination, combination_map_ on the master node is global combination map.
  bool glb_combine_ = true;  // Enable/disable global combination. (Enable by default.)

  /* Data Members Used for Time Sharing Mode Only */
  static const size_t CIRCULAR_BUF_SIZE_ = 10;  // The maximum # of cached time steps in the circular buf. 
  In* circular_buf_[CIRCULAR_BUF_SIZE_];  // The circular buffer to cache time steps.
  bool cell_flags_[CIRCULAR_BUF_SIZE_];  // Each flag indicates a cell is either allocated or empty.
  size_t p_idx_ = 0;  // Cell index in the buffer for the producer (simulation task).
  size_t c_idx_ = 0;  // Cell index in the buffer for the consumer (analytics task).
};

template <class In, class Out>
Scheduler<In, Out>::Scheduler(const SchedArgs& args) : num_threads_(args.num_threads), step_(args.step), extra_data_(args.extra_data), num_iters_(args.num_iters) {
  MPI_Comm_size(MPI_COMM_WORLD, &num_nodes_);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank_);

  if (rank_ == 0)
    printf("Scheduler: Initializing with %d threads and %d nodes...\n", args.num_threads, num_nodes_);
	assert(args.num_threads > 0);
  assert(args.step > 0);

  if (rank_ == 0)
	  printf("Scheduler: Constructing the reduction map for all the threads...\n"); 
  for (int i = 0; i < num_threads_; ++i) {
    map<int, unique_ptr<RedObj>> loc_reduction_map;
    reduction_map_.emplace_back(move(loc_reduction_map));
  }
  if (rank_ == 0)
    printf("Scheduler: Reduciton map for %lu threads is ready.\n", reduction_map_.size()); 

  // Extra initialization for time sharing mode.
  memset(cell_flags_, 0, CIRCULAR_BUF_SIZE_ * sizeof(bool));  // Each unset flag indicates an empty cell.
  memset(circular_buf_, 0, CIRCULAR_BUF_SIZE_ * sizeof(In*));  // No cell has been allocated in the buffer.
}

template <class In, class Out>
void Scheduler<In, Out>::setup(const In* data, size_t total_len, size_t buf_size, Out* out, size_t out_len) {
  assert(data != nullptr);
  data_ = data;
  assert(total_len > 0);
  total_len_ = total_len;
  assert(buf_size > 0 && buf_size <= total_len);
  buf_size_ = buf_size;
  if (out != nullptr) {
    assert(out_len > 0);
    out_ = out;
    out_len_ = out_len;
  }
}

template <class In, class Out>
void Scheduler<In, Out>::run_space_sharing(const In* data, size_t total_len, size_t buf_size, Out* out, size_t out_len, MAPPING_MODE_T mode) {
  //chrono::time_point<chrono::system_clock> clk_beg, clk_end;
  //clk_beg = chrono::system_clock::now();

  // Clear both reduction and combination maps.
  reduction_map_.clear();
  combination_map_.clear();

  // Set up the scheduler.
  setup(data, total_len, buf_size, out, out_len);
  // Process extra_data_ to help intialize combination_map_.
  process_extra_data();

  if (rank_ == 0) {
    dprintf("Combination map after processing extra data...\n");
    //dump_combination_map();
  }

  for (int iter = 1; iter <= num_iters_; ++iter) {
    //if (rank_ == 0)
    //  dprintf("Scheduler: Iteration# = %d.\n", iter);

    // Set the input data chunk.  
    Chunk input(0, buf_size_);

    // Copy global combination map to each (local) combination_map_.
    if (iter > 1) {
      distribute_global_combination_map();

      dprintf("Combination map after distributing global combination map on node %d...\n", rank_);
      //dump_combination_map();
    }

    // If process_extra_data is not overridden, then nothing will be done, since
    // combination_map_ is initally empty.
    // Such distribution is done in parallel.
    distribute_local_combination_map();

    dprintf("Reduction map after distributing local combination map on node %d...\n", rank_);
    //dump_reduction_map();

    // Process chunks one by one.
    // Mainly perform splitting and local reduction.
    int num_bufs_to_process = total_len_ / buf_size_;
    if (total_len_ % buf_size_ != 0) {
      ++num_bufs_to_process;
    }
    for (int i = 0; i < num_bufs_to_process; ++i) {
      dprintf("\nScheduler: Processing the input chunk %d...\n", i + 1);

      if (i != num_bufs_to_process - 1 || total_len_ % buf_size_ == 0) {
        process_chunk(input, mode);
        input.start += buf_size_;
      } else {  // The last iteration does not load a full buffer.
        // Set the input length as the actual length of the remaining data.
        input.length = total_len_ % buf_size_;
        process_chunk(input, mode);
      }
    }

    dprintf("Reduction map after local reduction...\n");
    //dump_reduction_map();

    // Local combination is done sequentially (or can be at most pairwise parallelized).
    local_combine(); 

    dprintf("Combination map after local combination...\n");
    //dump_combination_map();

    // Global combination is doen sequentially (or at most pairwise parallelized
    // and multi-threaded).
    if (glb_combine_)
      global_combine();

    dprintf("Combination map after global combination...\n");
    //dump_combination_map();

    // Perform post-combination processing on the master node.
    // It is meaningless to perform post-combination on slave nodes,
    // since global results are only maintained on the master node.
    if (rank_ == 0) {
      post_combine();

      dprintf("Global combination map after post-combination at iteration %d...\n", iter);
      //dump_combination_map();
    }
  }

  // Ouptut (local) combination_map_ to each node's output destination.
  // If the output destination is only valid for the master node,
  // then only the (global) combination_map_ on the master node will be output.
  if (out != nullptr && out_len > 0) {
    output();
  }

  //clk_end = chrono::system_clock::now();
  //chrono::duration<double> elapsed_seconds = clk_end - clk_beg;
  //if (num_iters_ > 1 && rank_ == 0) {
  //  dprintf("Scheduler: # of iterations = %d.\n", num_iters_);
  //}
 
  //dprintf("Scheduler: Processing time on node %d = %.2f secs.\n", rank_, elapsed_seconds.count());
}

template <class In, class Out>
void Scheduler<In, Out>::run_time_sharing(Out* out, size_t out_len, MAPPING_MODE_T mapping_mode) {
    // Wait for a produced cell.
    while(!cell_flags_[c_idx_]) {
      dprintf("Scheduler: Waiting for producing time steps...\n");
      sleep(1);
    }

    dprintf("Scheduler: Dequeue a new time step...\n");
    if (mapping_mode == GEN_ONE_KEY_PER_CHUNK)
      Scheduler<In, Out>::run(circular_buf_[c_idx_], total_len_, out, out_len);
    else  // mapping_mode = GEN_KEYS_PER_CHUNK.
      Scheduler<In, Out>::run2(circular_buf_[c_idx_], total_len_, out, out_len);

    cell_flags_[c_idx_] = false;
    c_idx_ = (c_idx_ + 1) % CIRCULAR_BUF_SIZE_;
}


template <class In, class Out>
void Scheduler<In, Out>::process_chunk(const Chunk& input, MAPPING_MODE_T mode) {
  if (input.empty())
    return;

  assert(input.start >= 0 && input.start + input.length <= total_len_);
  input_ = input;

  // Splitting (or local partitioning) is done by the master thread.
  split();

  #pragma omp parallel num_threads(num_threads_)
  {
    reduce(mode);
  }

  // Clear splits_ for the next run.
  // Such clearance is only required for iterative processing.
  splits_.clear();
}

template <class In, class Out>
void Scheduler<In, Out>::distribute_global_combination_map() {
  dprintf("Scheduler: Distribute global combination map to each (local) combination_map_.\n");

  int num_red_objs = 0;
  if (rank_ == 0) {
    num_red_objs = (int)combination_map_.size();
  }

  MPI_Bcast(&num_red_objs, 1, MPI_INT, 0, MPI_COMM_WORLD);
  dprintf("num_red_objs on node %d = %d\n", rank_, num_red_objs); 

  // Vectorize global comination map;
  int keys[num_red_objs];
  size_t length = num_red_objs * red_obj_size_;  // Total length of the serialized reduction objects.
  char *red_objs = new char[length];

  if (rank_ == 0) {
    int i = 0;
    for (const auto& pair : combination_map_) {
      keys[i] = pair.first;
      memcpy(&red_objs[i * red_obj_size_], pair.second.get(), red_obj_size_);
      ++i;
    }
  }

  // Broadcast vectorized combination_map_;
  MPI_Bcast(keys, num_red_objs, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(red_objs, length, MPI_BYTE, 0, MPI_COMM_WORLD);

  // Generate global combination map from received vectorized (local) combination_map_.
  if (rank_ != 0) {
      combination_map_.clear();

      for (int i = 0; i < num_red_objs; ++i) {
        deserialize(combination_map_[keys[i]], &red_objs[i * red_obj_size_]);
      }
  }

  dprintf("Scheduler: Combination map after distributing global combination map on node %d.\n", rank_);
  //dump_combination_map();

  delete [] red_objs;
}

template <class In, class Out>
void Scheduler<In, Out>::distribute_local_combination_map() {
  dprintf("Scheduler: Distribute combination_map_ to each local reduction map.\n");

  #pragma omp parallel num_threads(num_threads_)
  {
    int tid = omp_get_thread_num();
    reduction_map_[tid].clear();

    for (const auto& pair: combination_map_) {
      const auto key = pair.first;
      reduction_map_[tid][key].reset(combination_map_.find(key)->second->clone());
    }
  }
}

template <class In, class Out>
void Scheduler<In, Out>::split() {
  dprintf("Scheduler: Splitting the input data into %d splits...\n", num_threads_);

  size_t split_length = input_.length / num_threads_;
  for (int i = 0; i < num_threads_; ++i) {
    Chunk c;
    c.start = input_.start + i * split_length;
    c.length = split_length;
    splits_.emplace_back(c);
  }

  if (input_.length % num_threads_ != 0) {
    splits_.back().length += input_.length % num_threads_;
  }

  // Print out each split.
  //for (int i = 0; i < num_threads_; ++i) {
  //  printf("Split for thread %d: start = %lu, length = %lu.\n", i, splits_[i].start, splits_[i].length);
  //}
}

template <class In, class Out>
bool Scheduler<In, Out>::next(unique_ptr<Chunk>& unit_chunk, const Chunk& split) const {
  bool is_last = false;

  // Retrieve a non-first unit chunk.
  if (unit_chunk != nullptr) {
    unit_chunk->start += unit_chunk->length; 
    // The length equals step_, so remain unchanged for the non-last unit chunk.
    
    size_t split_end = split.start + split.length;
    if (unit_chunk->start + unit_chunk->length >= split_end) {
      unit_chunk->length = split_end - unit_chunk->start;
      is_last = true;
    }
  } else {  // Retrieve the first unit chunk.
    unit_chunk.reset(new Chunk);
    unit_chunk->start = split.start;

    if (step_ < split.length) {
      unit_chunk->length = step_;
    } else {
      unit_chunk->length = split.length;
      is_last = true;
    }
  }

  dprintf("Current unit chunk: start = %lu, length = %lu.\n", unit_chunk->start, unit_chunk->length);
  return is_last;
}

template <class In, class Out>
void Scheduler<In, Out>::reduce(MAPPING_MODE_T mode) {
  int tid = omp_get_thread_num(); 
  dprintf("Scheduler: Local reduction on thread %d on node %d...\n", tid, rank_);

  bool is_last = false;
  unique_ptr<Chunk> chunk = nullptr;
  do {
    is_last = next(chunk, splits_[tid]);

    if (mode == GEN_ONE_KEY_PER_CHUNK) {  // Perform reduction with gen_key.
      int key = gen_key(*chunk);
      accumulate(*chunk, reduction_map_[tid][key]);
      // Check if the early emission can be triggered.
      if (reduction_map_[tid].find(key)->second->trigger()) {
        dprintf("Scheduler: The reduction object %s is emitted by trigger...\n", reduction_map_[tid].find(key)->second->str().c_str());
        assert(key >= 0 && key < out_len_);
        convert(*reduction_map_[tid].find(key)->second, &out_[key]);
        reduction_map_[tid].erase(key);
      }
    } else { // mode == GEN_KEYS_PER_CHUNK, and perform reduction with gen_keys.
      vector<int> keys;
      gen_keys(*chunk, keys);

      for (int key : keys) {
        accumulate(*chunk, reduction_map_[tid][key]);
        // Check if the early emission can be triggered.
        if (reduction_map_[tid].find(key)->second->trigger()) {
          dprintf("Scheduler: The reduction object %s is emitted by trigger...\n", reduction_map_[tid].find(key)->second->str().c_str());
          assert(key >= 0 && key < out_len_);
          convert(*reduction_map_[tid].find(key)->second, &out_[key]);
          reduction_map_[tid].erase(key);
        }
      }
    }
  } while (!is_last);
}

// No parital concurrency during combination is explored for now.
template <class In, class Out>
void Scheduler<In, Out>::local_combine() {
  dprintf("Scheduler: Local combination on the master thread on node %d...\n", rank_); 

  // Combine all the local reduction maps with the combination map.
  for (int tid = 0; tid < num_threads_; ++tid) {
    for (auto& pair : reduction_map_[tid]) {
      if (combination_map_.find(pair.first) != combination_map_.end()) {
        merge(*pair.second, combination_map_[pair.first]);
      } else {
       combination_map_[pair.first] = move(pair.second);
      }
    }
  }
}

// No parital concurrency during combination is explored for now.
template <class In, class Out>
void Scheduler<In, Out>::global_combine() {
  dprintf("Scheduler: Global combination...\n"); 

  int num_nodes;
  MPI_Comm_size(MPI_COMM_WORLD, &num_nodes);
  int local_num_red_objs = (int)combination_map_.size();
  int global_num_red_objs[num_nodes];
  MPI_Gather(&local_num_red_objs, 1, MPI_INT, global_num_red_objs, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (rank_ == 0) {
    for (int i = 1; i < num_nodes; ++i) {
      // Receive serialized reduciton objects from slave nodes.
      int keys[global_num_red_objs[i]];
      size_t length = red_obj_size_ * global_num_red_objs[i];  // Total length of the serialized reduction objects.
      char* red_objs = new char[red_obj_size_ * global_num_red_objs[i]];

      MPI_Status status;
      MPI_Recv(keys, global_num_red_objs[i], MPI_INT, i, i, MPI_COMM_WORLD, &status);
      MPI_Recv(red_objs, length, MPI_BYTE, i, i + num_nodes, MPI_COMM_WORLD, &status);

      // Deserialize reduction objects and add them to the global combination
      // map.
      for (int j = 0; j < global_num_red_objs[i]; ++j) {
        unique_ptr<RedObj> red_obj;
        deserialize(red_obj, &red_objs[j * red_obj_size_]);
        assert(red_obj != nullptr);

        if (combination_map_.find(keys[j]) != combination_map_.end()) {
          merge(*red_obj, combination_map_[keys[j]]); 
        } else {
          combination_map_[keys[j]] = move(red_obj);
        } 
      }

      delete [] red_objs;
    }
  } else {
    // Serialize reduction objects.
    int i = 0;
    int local_keys[local_num_red_objs];
    size_t length = red_obj_size_ * local_num_red_objs;  // Total length of the serialized reduction objects.
    char* local_red_objs = new char[length];

    for (const auto& pair : combination_map_) {
      local_keys[i] = pair.first;
      memcpy(&local_red_objs[i * red_obj_size_], pair.second.get(), red_obj_size_);
      ++i;
    }

    // Send serialized data to the master node.
    MPI_Send(local_keys, local_num_red_objs, MPI_INT, 0, rank_, MPI_COMM_WORLD);
    MPI_Send(local_red_objs, length, MPI_BYTE, 0, rank_ + num_nodes, MPI_COMM_WORLD);
  }
}

template <class In, class Out>
void Scheduler<In, Out>::output() {
  for (const auto& pair: combination_map_) {
    int key = pair.first;
    assert(key >= 0 && key < out_len_);

    // Convert a reduction object into a desired output element.
    convert(*pair.second.get(), &out_[key]);
  } 
}

template <class In, class Out>
void Scheduler<In, Out>::feed(const In* data, size_t total_len) {
  // Wait for an empty cell.
  while(cell_flags_[p_idx_]) {
    dprintf("Scheduler: Waiting for consuming time steps...\n");
    sleep(1);
  }

  // Set or validate the length of the input.
  // Assume all the time steps have the same length.
  // Otherwise, another vector of varying total lengths should be maintained.
  if (total_len_ == 0) {
    total_len_ = total_len;
  } else {
    assert(total_len_ == total_len);
  }

  dprintf("Scheduler: Enqueue a new time step...\n");
  if (circular_buf_[p_idx_] == nullptr)
    circular_buf_[p_idx_] = new In[total_len];
  memcpy(circular_buf_[p_idx_], data, total_len * sizeof(In));

  cell_flags_[p_idx_] = true;
  p_idx_ = (p_idx_ + 1) % CIRCULAR_BUF_SIZE_; 
}

template <class In, class Out>
void Scheduler<In, Out>::dump_reduction_map() const {
	printf("Reduction map on node %d:\n", rank_);
	for (int i = 0; i < num_threads_; ++i) {
		printf("\tLocal reduciton map %d:\n", i);
		for (const auto& pair : reduction_map_[i]) {
			printf("\t\t(key = %d, value = %s)\n", pair.first, (pair.second != nullptr ? pair.second->str().c_str() : "NULL"));
		}
	}
}

template <class In, class Out>
void Scheduler<In, Out>::dump_combination_map() const {
	printf("Combination map on node %d:\n", rank_);
	for (const auto& pair : combination_map_) {
		printf("\t(key = %d, value = %s)\n", pair.first, (pair.second != nullptr ? pair.second->str().c_str() : "NULL"));
	}
}

#endif  // _SCHEDULER_H_
