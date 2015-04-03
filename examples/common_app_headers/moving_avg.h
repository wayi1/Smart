#ifndef	_MOVING_AVERAGE_H_
#define	_MOVING_AVERAGE_H_

#include <algorithm>
#include <cmath>
#include <memory>

#include "chunk.h"
#include "scheduler.h"

using namespace std;

#define RADIUS 3  // Actually for 1D array, window size = radius * 2 + 1. 
#define WIN_SIZE  2 * RADIUS + 1  // Window size.

// Sliding window object. 
struct WinObj : public RedObj {
  double sum = 0;
  size_t count = 0;

  // Optional, only used for rendering the result.
  string str() const override {
    return "(sum = " + to_string(sum) + ", count = " + to_string(count) + ")";
  }

  // Trigger early emission when the count reaches WIN_SIZE.
  bool trigger() const override {
    return count >= WIN_SIZE;
  }
};

template <class In, class Out>
class MovingAverage : public Scheduler<In, Out> {
 public:
  using Scheduler<In, Out>::Scheduler;

  // Group elements into buckets.
  void gen_keys(const Chunk& chunk, vector<int>& keys) const override {
    dprintf("For chunk[%lu], genrate key %lu from node%d...\n", chunk.start, chunk.start, this->rank_);
    keys.emplace_back(chunk.start);
    // Assume that chunk.start + RADIUS will be within the size_t value
    // range.
    for (size_t i = chunk.start + 1; i <= min(chunk.start + RADIUS, this->total_len_ - 1); ++i) {
      dprintf("For chunk[%lu], genrate key %lu from node%d...\n", chunk.start, i, this->rank_);
      keys.emplace_back(i);
    }

    // Note that since the type size_t is unsigned,
    // chunk.start - RADIUS can be a large positive value than a negative value.
    if (chunk.start >= RADIUS) { 
      for (size_t i = chunk.start - RADIUS; i < chunk.start; ++i) {
        dprintf("For chunk[%lu], genrate key %lu from node%d...\n", chunk.start, i, this->rank_);
        keys.emplace_back(i);
      }
    } else {
      for (size_t i = 0; i < chunk.start; ++i) {
        dprintf("For chunk[%lu], genrate key %lu from node%d...\n", chunk.start, i, this->rank_);
        keys.emplace_back(i);
      }
    }
  }

  // Accumulate sum and count.
  void accumulate(const Chunk& chunk, unique_ptr<RedObj>& red_obj) override {
    if (red_obj == nullptr) {
      red_obj.reset(new WinObj);
    }

    WinObj* w = static_cast<WinObj*>(red_obj.get());  
    for (size_t i = 0; i < chunk.length; ++i) {
      dprintf("Adding the element chunk[%lu] = %.0f.\n", chunk.start + i, this->data_[chunk.start + i]);
      w->sum += (double)this->data_[chunk.start + i];
      w->count++;
    }

    dprintf("After local reduction, w = %s.\n", w->str().c_str()); 
  }

  // Merge sum and count.
  void merge(const RedObj& red_obj, unique_ptr<RedObj>& com_obj) override {
    const WinObj* wr = static_cast<const WinObj*>(&red_obj);
    WinObj* wc = static_cast<WinObj*>(com_obj.get());

    wc->sum += wr->sum;
    wc->count += wr->count;
  }

  // Deserialize reduction object. 
  void deserialize(unique_ptr<RedObj>& obj, const char* data) const override {
    obj.reset(new WinObj);
    memcpy(obj.get(), data, sizeof(WinObj));
  }

  // Convert a reduction object into a desired output element.
  void convert(const RedObj& red_obj, Out* out) const override {
    const WinObj* w = static_cast<const WinObj*>(&red_obj);
    *out = w->sum / w->count;
  }
};

#endif  // _MOVING_AVERAGE_H_
