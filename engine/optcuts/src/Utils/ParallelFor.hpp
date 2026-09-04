#pragma once

#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>

namespace uvgami {

// a tbb task per triangle costs more than the triangle in a local query solve
const int PARALLEL_LOOP_MIN_ITEMS = 64;
const int PARALLEL_LOOP_GRAIN = 16;

template <class Body> void parallelFor(int amount, const Body &body) {
    if (amount < PARALLEL_LOOP_MIN_ITEMS) {
        for (int i = 0; i < amount; ++i)
            body(i);
        return;
    }
    tbb::parallel_for(tbb::blocked_range<int>(0, amount, PARALLEL_LOOP_GRAIN),
                      [&](const tbb::blocked_range<int> &range) {
                          for (int i = range.begin(); i != range.end(); ++i)
                              body(i);
                      });
}

// min or max reductions only, the fold order changes with the split
template <class Body, class Join>
double parallelReduce(int amount, double init, const Body &body,
                      const Join &join) {
    if (amount < PARALLEL_LOOP_MIN_ITEMS)
        return body(tbb::blocked_range<int>(0, amount), init);
    return tbb::parallel_reduce(
        tbb::blocked_range<int>(0, amount, PARALLEL_LOOP_GRAIN), init, body,
        join);
}

} // namespace uvgami
