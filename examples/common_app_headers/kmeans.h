#ifndef	_KMEANS_H_
#define	_KMEANS_H_

#include <cassert>
#include <cfloat>
#include <cmath>
#include <cstring>
#include <memory>

#include "chunk.h"
#include "scheduler.h"

using namespace std;

#define NUM_DIMS  8  // The # of dimensions for a point.
#define NUM_MEANS 4  // The # of clusters/centroids.

#define OBSERVE_CLUSTER_SIZE  0  // Need to obeserve the change of cluster sizes?

// Cluster assoicated data.
template <class T>
struct ClusterObj : public RedObj {
  T centroid[NUM_DIMS];  // The centroid. After a cluster is formed, centroid[i] = sum[i] / size;
  T sum[NUM_DIMS];  // The sum of coordinate value in each dimension for each point in the cluster.
  size_t size = 0;  // The # of points in the cluster.

  ClusterObj() {
    memset(sum, 0, NUM_DIMS * sizeof(T));
  }

  ClusterObj(const ClusterObj& obj) {
    memcpy(this, &obj, sizeof(ClusterObj));
  }

  // Required only if distributing combination map is critical.
  RedObj* clone() override {
    return new ClusterObj(*this);
  }

  // Optional, only used for rendering the result.
  string str() const override {
    string str = "(centroid = [";
    for (int i = 0; i < NUM_DIMS - 1; ++i) {
      str += to_string(centroid[i]) + ", ";
    } 
    str += to_string(centroid[NUM_DIMS - 1]) + "], sum = [";
    for (int i = 0; i < NUM_DIMS - 1; ++i) {
      str += to_string(sum[i]) + ", ";
    } 
    str += to_string(sum[NUM_DIMS - 1]) + "], size = " + to_string(size) + ")";
    return str;
  }

  /* Additional Functions */
  // Update the centroid with the sum and size.
  void update_centroid() {
    for (int i = 0; i < NUM_DIMS; ++i) {
      centroid[i] = sum[i] / size;
    }
  }

  // Clear the belonged points. Only keep the centroid.
  void clear() {
    memset(sum, 0, NUM_DIMS * sizeof(T));
    size = 0;
  }

  // Retrieve the squared distance between a specified centroid and a given point.
  static T inline sq_dist(const T* centroid, const T* point) {
    T sq_dist = 0;
    for (int i = 0; i < NUM_DIMS; ++i) {
      sq_dist += pow(centroid[i] - point[i], 2);
    }

    return sq_dist;
  }
};

template <class T>
class Kmeans : public Scheduler<T, T*> {
 public:
  using Scheduler<T, T*>::Scheduler;

  // Each chunk is viewed as a multi-dimensional point.
  // Identify the cluster index given the point and all the current centroids.
  int gen_key(const Chunk& chunk) const override {
    if (chunk.empty())
      return NAN;

    assert(chunk.length == NUM_DIMS);

    T min_sq_dist = (T)DBL_MAX;
    T cur_sq_dist = (T)DBL_MAX;
    int cluster_id = -1;

    for (int i = 0; i < NUM_MEANS; ++i) {
      const ClusterObj<T>* cur_cluster_obj = static_cast<const ClusterObj<T>*>(this->combination_map_.find(i)->second.get());
      const T* centroid = cur_cluster_obj->centroid;

      cur_sq_dist = ClusterObj<T>::sq_dist(centroid, &this->data_[chunk.start]);
      if (cur_sq_dist < min_sq_dist) {
        min_sq_dist = cur_sq_dist;
        cluster_id = i;
      }
    }

    dprintf("cluster_id = %d, min_sq_dist = %.2f.\n", cluster_id, (float)min_sq_dist);
    return cluster_id;
  }

  // Accumulate sum and size.
  void accumulate(const Chunk& chunk, unique_ptr<RedObj>& red_obj) override {
    if (chunk.empty())
      return;

    assert(chunk.length == NUM_DIMS);
    assert(red_obj);

    ClusterObj<T>* cluster_obj = static_cast<ClusterObj<T>*>(red_obj.get());
    dprintf("chunk.start = %lu, cluster_obj = %s.\n", chunk.start, cluster_obj->str().c_str());
    for (int i = 0; i < NUM_DIMS; ++i) {
      cluster_obj->sum[i] += this->data_[chunk.start + i];
    }
    cluster_obj->size++;
    dprintf("After local reduction, cluster_obj = %s.\n", cluster_obj->str().c_str());
  }

  // Merge sum and size.
  void merge(const RedObj& red_obj, unique_ptr<RedObj>& com_obj) override {
    const ClusterObj<T>* red_cluster_obj = static_cast<const ClusterObj<T>*>(&red_obj);
    ClusterObj<T>* com_cluster_obj = static_cast<ClusterObj<T>*>(com_obj.get());

    for (int i = 0; i < NUM_DIMS; ++i) {
      com_cluster_obj->sum[i] += red_cluster_obj->sum[i];
    }
    com_cluster_obj->size += red_cluster_obj->size;
  }

  // Deserialize reduction object. 
  void deserialize(unique_ptr<RedObj>& obj, const char* data) const override {
    obj.reset(new ClusterObj<T>);
    memcpy(obj.get(), data, sizeof(ClusterObj<T>));
  }

  // Convert a reduction object into a desired output element.
  void convert(const RedObj& red_obj, T** out) const override {
    const ClusterObj<T>* cluster_obj = static_cast<const ClusterObj<T>*>(&red_obj);
    memcpy(*out, cluster_obj->centroid, sizeof(T) * NUM_DIMS);
  }

  /* Additional Function Overriding */
  // Set up the initial centroids in combination_map_.
  void process_extra_data() override {
    dprintf("Scheduler: Processing extra data...\n");

    assert(this->extra_data_ != nullptr);
    const T** centroids = (const T**)this->extra_data_;
    for (int i = 0; i < NUM_MEANS; ++i) {
      // Initialize the result cluster with the initial centroids
      unique_ptr<ClusterObj<T>> cluster_obj(new ClusterObj<T>);
      memcpy(cluster_obj->centroid, centroids[i], NUM_DIMS * sizeof(T));

      // Update combination_map_.
      this->combination_map_[i] = move(cluster_obj);
      dprintf("combination_map_[%d] = %s\n", i, this->combination_map_[i]->str().c_str());
    }
  }

  // Finalize combinaion_map_.
  void post_combine() override {
    if (!OBSERVE_CLUSTER_SIZE) {
      // Process update and clearance altogether.
      for (auto& pair : this->combination_map_) {
        ClusterObj<T>* cluster_obj = static_cast<ClusterObj<T>*>(pair.second.get());         
        // Update the centroids for each cluster.
        cluster_obj->update_centroid();

        // Clear sum and size in the cluster object.
        cluster_obj->clear();
      }
    } else {  // Process update and clearance in separate loops to observe the change of cluster sizes.
      for (auto& pair : this->combination_map_) {
        ClusterObj<T>* cluster_obj = static_cast<ClusterObj<T>*>(pair.second.get());         
        // Update the centroids for each cluster.
        cluster_obj->update_centroid();
      }
      printf("Local combination map before cluster object clearance.\n");
      this->dump_combination_map();

      for (auto& pair : this->combination_map_) {
        ClusterObj<T>* cluster_obj = static_cast<ClusterObj<T>*>(pair.second.get());         
        // Clear sum and size in the cluster object.
        cluster_obj->clear();
      }
    }
  } 
};

#endif  // _KMEANS_H_
