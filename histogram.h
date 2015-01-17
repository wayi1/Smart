#ifndef	_HISTOGRAM_H_
#define	_HISTOGRAM_H_

#include <cmath>

#include "scheduler.h"

using namespace std;

#define BUCKET_WIDTH  100 // Bucket width. Make sure the value of STEP in simulation code is 1.
#define MIN_VAL 0 // The minimum value in the value domain. 

// Histogram bucket.
struct Hist : public RedObj {
  size_t count = 0;

  // Optional, only used for rendering the result.
  string str() const override {
    return "(count = " + to_string(count) + ")";
  }
};

template <class In>
class Histogram : public Scheduler<In, size_t> {
public:
  using Scheduler<In, size_t>::Scheduler;

  // Group elements into buckets.
  int gen_key(const Chunk& chunk) const override {
      return (int)(this->data_[chunk.start] - MIN_VAL) / BUCKET_WIDTH;
  }

  // Acumulate sum and count.
  void accumulate(const Chunk& chunk, unique_ptr<RedObj>& red_obj) override {
    if (red_obj == nullptr) {
      red_obj.reset(new Hist);
    }

    Hist* h = static_cast<Hist*>(red_obj.get());  
    for (size_t i = 0; i < chunk.length; ++i) {
      dprintf("Adding the element chunk[%lu] = %.0f.\n", chunk.start + i, this->data_[chunk.start + i]);
      h->count++;
    }

    dprintf("After local reduction, h = %s.\n", h->str().c_str()); 
  }

  // Merge sum and count.
	void merge(const RedObj& red_obj, unique_ptr<RedObj>& com_obj) override {
    const Hist* hr = static_cast<const Hist*>(&red_obj);
    Hist* hc = static_cast<Hist*>(com_obj.get());

    hc->count += hr->count;
  }

	// Deserialize reduction object. 
  void deserialize(unique_ptr<RedObj>& obj, const char* data) const override {
		obj.reset(new Hist);
	  memcpy(obj.get(), data, sizeof(Hist));
  }

  // Convert a reduction object into a desired output element.
  //void convert(const RedObj& red_obj, size_t* out) const override {
  //  const Hist* h = static_cast<const Hist*>(&red_obj);
  //  *out = h->count;
  //}
};

#endif	// _HISTOGRAM_H_
