#ifndef	_LR_H_
#define	_LR_H_

#include <cassert>
#include <cmath>
#include <cstring>
#include <map>
#include <memory>

#include "chunk.h"
#include "scheduler.h"

using namespace std;

#define NUM_DIMS  15  // The # of dimensions for a the explaining variable x.
#define NUM_COLS  NUM_DIMS + 1  // The # of dimensions for x plus a unary target variable y.

// Print a feature vector.
void printVector(const double v[]) {
  printf("[");
  for (int i = 0; i < NUM_DIMS - 1; ++i) {
    printf("%.17f, ", v[i]);
  }
  printf("%.17f]\n", v[NUM_DIMS - 1]);
}

// Compute the dot product of two feature vectors.
double inline dot_product(const double a[], const double b[]) {
  double dot = 0;
  for (int i = 0; i < NUM_DIMS; ++i) {
    dot += a[i] * b[i];
  }
  return dot;
}

// Note that the length of point is NUM_COLS, and the length of weights is NUM_DIMS.
// The value of the target variable is point[NUM_DIMS].
void inline compute(const double point[], const double weights[], double unit_gradient[]) {
  double dot = dot_product(point, weights);
  for (int i = 0; i < NUM_DIMS; ++i) {
    unit_gradient[i] = (1 / (1 + exp(-1 * point[NUM_DIMS] * dot)) - 1) * point[NUM_DIMS] * point[i];
  }
}

// Add a feature vector to another feature vector.
void inline vector_sum(const double v[], double sum[]) {
  for (int i = 0; i < NUM_DIMS; ++i) {
    sum[i] += v[i];
  }
}

// Substract a feature vector from another feature vector.
void inline vector_diff(const double v[], double diff[]) {
  for (int i = 0; i < NUM_DIMS; ++i) {
    diff[i] -= v[i];
  }
}

// Gradient data.
struct GradientObj : public RedObj {
  double gradient[NUM_DIMS];
  double weight[NUM_DIMS];

  GradientObj() {
    memset(gradient, 0, NUM_DIMS * sizeof(double));
  }

  GradientObj(const GradientObj& g) {
    memcpy(this, &g, sizeof(GradientObj));
  }

  // Required only if distribute_combination_map() is critical.
  GradientObj* clone() override {
    return new GradientObj(*this);
  }

  // Optional, only used for rendering the result.
  string str() const override {
    string str = "(gradient = [";
    for (int i = 0; i < NUM_DIMS - 1; ++i) {
      str += to_string(gradient[i]) + ", ";
    } 
    str += to_string(gradient[NUM_DIMS - 1]) + "], weight = [";
    for (int i = 0; i < NUM_DIMS - 1; ++i) {
      str += to_string(weight[i]) + ", ";
    } 
    str += to_string(weight[NUM_DIMS - 1]) + "])";
    return str;
  }

  /* Additional Functions*/
  // Add a unit gradient to the local total gradient.
  void update_gradient(const double unit_gradient[]) {
    vector_sum(unit_gradient, gradient);
  }

  // Substract the gradient from the weight.
  void update_weight() {
    vector_diff(gradient, weight);
  }

  // Clear the belonged gradient. Only keep the weight.
  void clear() {
    memset(gradient, 0, NUM_DIMS * sizeof(double));
  }
};

template <class In>
class LogisticRegression : public Scheduler<In, double*> {
 public: 
  using Scheduler<In, double*>::Scheduler;

  // A unique key value that will be stored in reduction/combination map.
  const static int UNIQUE_KEY = 0;

  // Each chunk is viewed as a combination of explaining variable and target
  // variable, and the total lenght is NUM_COLS = NUM_DIMS + 1.
  // Since there is only one global gradient to maintain, return a unique and fixed
  // integer in this case.
  int gen_key(const Chunk& chunk, const In* data, map<int, unique_ptr<RedObj>>& combination_map) const override {
    if (chunk.empty())
      return NAN;

    assert(chunk.length == NUM_COLS);

    // Only a unique fixed key is needed.
    return UNIQUE_KEY;
  }

  // Accumulate gradient. 
  void accumulate(const Chunk& chunk, const In* data, unique_ptr<RedObj>& red_obj) override {
    if (chunk.empty())
      return;

    assert(chunk.length == NUM_COLS);
    assert(red_obj);

    GradientObj* g_obj = static_cast<GradientObj*>(red_obj.get());
    dprintf("chunk.start = %lu, g_obj = %s.\n", chunk.start, g_obj->str().c_str());

    // Compute the (unit) gradient for the given chunk.
    double unit_gradient[NUM_DIMS];
    compute(&data[chunk.start], g_obj->weight, unit_gradient);

    // Accumulte unit gradient to the local total gradient.
    g_obj->update_gradient(unit_gradient);

    dprintf("After local reduction, gradient = %s.\n", g_obj->str().c_str());
  }

  // Merge gradient.
  void merge(const RedObj& red_obj, unique_ptr<RedObj>& com_obj) override { 
    const GradientObj* red_g_obj = static_cast<const GradientObj*>(&red_obj);
    GradientObj* com_g_obj = static_cast<GradientObj*>(com_obj.get());
    vector_sum(red_g_obj->gradient, com_g_obj->gradient);
  }

  // Deserialize reduction object. 
  void deserialize(unique_ptr<RedObj>& obj, const char* data) const override {
    obj.reset(new GradientObj);
    memcpy(obj.get(), data, sizeof(GradientObj));
  }

  // Convert a reduction object into a desired output element.
  void convert(const RedObj& red_obj, double** out) const override {
    const GradientObj* g_obj = static_cast<const GradientObj*>(&red_obj);
    memcpy(*out, g_obj->weight, sizeof(double) * NUM_DIMS);
  }

  /* Additional Function Overriding */
  // Set up the initial weight in combination_map.
  void process_extra_data(const void* extra_data, map<int, unique_ptr<RedObj>>& combination_map) override {
    dprintf("Scheduler: Processing extra data...\n");

    assert(extra_data != nullptr);
    const double* weight = (const double*)extra_data;

    // Initialize the result gradient object with the initial weight.
    unique_ptr<GradientObj> g_obj(new GradientObj);
    memcpy(g_obj->weight, weight, NUM_DIMS * sizeof(double));

    // Update combination_map.
    combination_map[UNIQUE_KEY] = move(g_obj);
    dprintf("combination_map[%d] = %s\n", UNIQUE_KEY, combination_map[UNIQUE_KEY]->str().c_str());
  }

  // Finalize combinaion_map.
  void post_combine(map<int, unique_ptr<RedObj>>& combination_map) override {
    // There should be only one key-value pair is stored in the combination map.
    assert(combination_map.size() == 1);

    // Process update and clearance altogether.
    GradientObj* g_obj = static_cast<GradientObj*>(combination_map.find(UNIQUE_KEY)->second.get());         
    // Update the weight in the gradient object.
    g_obj->update_weight();
    // Clear gradient in the gradient object.
    g_obj->clear();
  }
};

#endif  // _LR_H_
