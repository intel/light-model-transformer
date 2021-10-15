// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifndef __TIMER_H_
#define __TIMER_H_
#include <sys/time.h>
#include <string>
#include <iostream>

using namespace std;

class Timer {
public:
  Timer(const char *_name) : name(_name), pTotalTime(NULL) {
    gettimeofday(&start, NULL);
  }

  Timer(const char *_name, float *_pTotal) : name(_name), pTotalTime(_pTotal) {
    gettimeofday(&start, NULL);
  }

  ~Timer() {
    gettimeofday(&end, NULL);
    float interval = (end.tv_sec - start.tv_sec) * 1000 + (end.tv_usec - start.tv_usec) / 1000.0f;
    if (pTotalTime) {
      *pTotalTime += interval;
    }
    cout << name << ": " << interval << " ms" << endl;
  }

private:
  string name;
  struct timeval start;
  struct timeval end;
  float *pTotalTime;
};

#endif
