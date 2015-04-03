#ifndef	_MOVING_MEDIAN_H_
#define	_MOVING_MEDIAN_H_

#include <algorithm>
#include <cmath>
#include <memory>
#include <sstream>
#include <vector>

#include "chunk.h"
#include "scheduler.h"

using namespace std;

#define RADIUS 1  // Actually for 1D array, window size = radius * 2 + 1. 
#define WIN_SIZE  2 * RADIUS + 1  // Window size.

// Sliding window object. 
struct WinObj : public RedObj {
  double win[WIN_SIZE];
  size_t count = 0;

  // Optional, only used for rendering the result.
  string str() const override {
    stringstream ss;
    ss << "(win = [";
    for (size_t i = 0; i < count - 1; ++i) {
      ss << win[i] << " ";
    }
    ss << win[count - 1];

    return "(win = [" + ss.str() + "], count = " + to_string(count) + ")";
  }

  // Trigger early emission when the count reaches WIN_SIZE.
  bool trigger() const override {
    return count >= WIN_SIZE;
  }
};

template <class In, class Out>
class MovingMedian : public Scheduler<In, Out> {
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

  // Accumulate the window.
  void accumulate(const Chunk& chunk, unique_ptr<RedObj>& red_obj) override {
    if (red_obj == nullptr) {
      red_obj.reset(new WinObj);
    }

    WinObj* w = static_cast<WinObj*>(red_obj.get());  
    assert(w->count + chunk.length <= WIN_SIZE);
    for (size_t i = 0; i < chunk.length; ++i) {
      dprintf("Adding the element chunk[%lu] = %.0f.\n", chunk.start + i, this->data_[chunk.start + i]);
      w->win[w->count++] = (double)this->data_[chunk.start + i];
    }

    dprintf("After local reduction, w = %s.\n", w->str().c_str()); 
  }

  // Merge the two windows.
  void merge(const RedObj& red_obj, unique_ptr<RedObj>& com_obj) override {
    const WinObj* wr = static_cast<const WinObj*>(&red_obj);
    WinObj* wc = static_cast<WinObj*>(com_obj.get());

    assert(wr->count + wc->count <= WIN_SIZE);
    for (size_t i = 0; i < wr->count; ++i) {
      wc->win[wc->count++] = wr->win[i];
    }
  }

  // Deserialize reduction object. 
  void deserialize(unique_ptr<RedObj>& obj, const char* data) const override {
    obj.reset(new WinObj);
    memcpy(obj.get(), data, sizeof(WinObj));
  }

  // Convert a reduction object into a desired output element.
  void convert(const RedObj& red_obj, Out* out) const override {
    const WinObj* w = static_cast<const WinObj*>(&red_obj);

    vector<double> v;
    v.assign(w->win, w->win + w->count);
    sort(v.begin(), v.end());
    if (w->count % 2 != 0) {
      *out = v[w->count / 2];
    } else {
      *out = (v[w->count / 2 - 1] + v[w->count / 2]) / 2;
    }
  }
};

#endif  // _MOVING_MEDIAN_H_
