#include <assert.h>
#include <stdlib.h>
#include <stdio.h>

#include <time.h>

#include "benes_router.h"

int packet_cmp(const void* a, const void* b) {
  data_t* pa = &(((packet_t*)a)->data);
  data_t* pb = &(((packet_t*)b)->data);
  if (pa->addr == pb->addr) {
    return (pa->timestamp < pb->timestamp) ? -1 : 1;
  } else {
    return (pa->addr < pb->addr) ? -1 : 1;
  }
}

int src_cmp(const void* a, const void* b) {
  packet_t* pa = ((packet_t*)a);
  packet_t* pb = ((packet_t*)b);
  return (pa->src < pb->src) ? -1 : 1;
}

void sort_packet(data_t* input, packet_t* packets, size_t n) {
  for (size_t i = 0; i < n; i++) {
    packets[i].data = input[i];
    packets[i].src = i;
  }
  qsort(packets, n, sizeof(packet_t), packet_cmp);
  for (size_t i = 0; i < n; i++) {
    packets[i].dst = i;
  }
  qsort(packets, n, sizeof(packet_t), src_cmp);
}

void route(packet_t* src_p, packet_t* b_dst_p, size_t n, size_t total_size) {
  packet_t* dst_p = src_p + total_size;
  packet_t* b_src_p = b_dst_p - total_size;
  if (n == 2) {
    assert(src_p == b_src_p);
    assert(dst_p == b_dst_p);
    // base condition
    //printf("src_p[0].dst=%ld\tsrc_p[1].dst=%ld\tn=%d\n", src_p[0].dst, src_p[1].dst, n);
    if (((src_p[0].dst & 0x1) == 0) && ((src_p[1].dst & 0x1) == 1)) {
      //printf("w: no switch\n");
      dst_p[0] = src_p[0];
      dst_p[1] = src_p[1];
    } else if (((src_p[0].dst & 0x1) == 1) && ((src_p[1].dst & 0x1) == 0)) {
      //printf("w: switch\n");
      dst_p[0] = src_p[1];
      dst_p[1] = src_p[0];
    } else {
      // something is wrong
      fprintf(stderr, "something is wrong.\n");
    }
    return;
  }

  size_t w = 0, w1, w2;
  while (true) {
    // only look at the least significant log(n) bits.
    w = w & (n - 1);
    //printf("w=%ld\tsrc_p[w].src=%ld\tn=%d\n", w, src_p[w].src, n);
    assert((src_p[w].src & (n - 1)) == w);

    if (src_p[w].routed) {
      // select the next one that is not routed.
      for (size_t i = 0; i < n; i++) {
        if (!src_p[i].routed) {
          w = i;
          break;
        }
      }
      if (src_p[w].routed) {
        // all packets in this level routed.
        break;
      }
    }

    w1 = src_p[w].dst & (n - 1);
    w2 = b_dst_p[w1 ^ (n >> 1)].src & (n - 1);
    //printf("w=%ld\tw1=%ld\tw2=%ld\n", w, w1, w2);

    // step (b)
    // forward routing, from src_p to dst_p
    //printf("w & (n >> 1)=%ld\tw ^ (n >> 1)=%ld\n", w & (n >> 1), w ^ (n >> 1));
    if (w & (n >> 1)) {
      //printf("w: switch\n");
      dst_p[w ^ (n >> 1)] = src_p[w];
    } else {
      //printf("w: no switch\n");
      dst_p[w] = src_p[w];
    }
    src_p[w].routed = true;

    // backward routing, from b_dst_p to b_src_p
    // route the pair
    if (w1 & (n >> 1)) {
      //printf("w1: switch\n");
      b_src_p[w1] = b_dst_p[w1 ^ (n >> 1)];
      b_src_p[w1 ^ (n >> 1)] = b_dst_p[w1];
    } else {
      //printf("w1: no switch\n");
      b_src_p[w1] = b_dst_p[w1];
      b_src_p[w1 ^ (n >> 1)] = b_dst_p[w1 ^ (n >> 1)];
    }

    // step (c)
    // step (d)
    if (w2 & (n >> 1)) {
      //printf("w2: no switch\n");
      dst_p[w2] = src_p[w2];
    } else {
      //printf("w2: switch\n");
      dst_p[w2 ^ (n >> 1)] = src_p[w2];
    }
    src_p[w2].routed = true;

    // Repeat this process until all packets are rounted in this layer.
    w = w2 ^ (n >> 1);
  }

  //for (size_t i = 0; i < n; i++) {
    //printf("%d\t%d\n", src_p[i].dst, dst_p[i].dst);
  //}
  //printf("=================================\n");
  //for (size_t i = 0; i < n; i++) {
    //printf("%d\t%d\n", b_src_p[i].dst, b_dst_p[i].dst);
  //}
  // recursive invocation to construct lower levels of the network.
  route(dst_p, b_src_p, n >> 1, total_size);
  route(dst_p + (n >> 1), b_src_p + (n >> 1), n >> 1, total_size);
}

bool data_equal(data_t* data1, data_t* data2) {
  return data1->addr == data2->addr && data1->timestamp == data2->timestamp && data1->type == data2->type && data1->value == data2->value;
}

void do_route(data_t* input, data_t* intermediate, data_t* output, switch_t* switches, size_t width, size_t depth) {
  if (width < 2) {
    printf("Benes network has to be at least of width 2\n");
    return;
  }
  packet_t* packets = new packet_t[width];

  for (size_t i = 0; i < width; i++) {
    packets[i].data = input[i];
    packets[i].routed = false;
  }

  sort_packet(input, packets, width);

  //for (size_t i = 0; i < width; i++) {
    //printf("data: %d,%d\tsrc: %ld\tdst: %ld\n", packets[i].data.addr, packets[i].data.timestamp, packets[i].src, packets[i].dst);
  //}

  packet_t* network = new packet_t[width * (depth + 1)];

  for (size_t i = 0; i < width; i++) {
    packets[i].routed = false;
    network[i] = packets[i];
    network[width * depth + packets[i].dst] = packets[i];
  }

  route(&network[0], &network[width * depth], width, width);

  // assign intermediate nodes, output, and switches.
  for (size_t i = 0; i < depth - 1; i++) {
    for (size_t j = 0; j < width; j++) {
      intermediate[i * width + j] = network[(i + 1) * width + j].data;
    }
  }
  for (size_t i = 0; i < width; i++) {
    output[i] = network[width * depth + i].data;
  }

  size_t level = (depth + 1) / 2;
  //assert(level == (int)(log((double)width)/log(2.0)));
  assert(depth == 2 * level - 1);

  // check the routing is done correctly.
  // first half of the benes network
  for (size_t i = 0; i < level; i++) {
    size_t gap_size = width >> (i + 1);
    size_t group_size = width >> i;
    size_t switches_in_a_group = group_size / 2;

    for (size_t j = 0; j < width; j++) {
      size_t group_id = j >> (level - i);
      size_t in_group_offset = j & (switches_in_a_group - 1);

      size_t target_index1 = (i + 1) * width + j;
      size_t source_index1 = target_index1 - width;
      size_t source_index2 = source_index1 ^ gap_size;
      size_t target_index2 = source_index2 + width;
      size_t switch_index = i * width / 2 + group_id * switches_in_a_group + in_group_offset;

      assert((data_equal(&network[target_index1].data,
              &network[source_index1].data) &&
            data_equal(&network[target_index2].data,
              &network[source_index2].data)) ||
          (data_equal(&network[target_index1].data,
                      &network[source_index2].data) &&
           data_equal(&network[target_index2].data,
             &network[source_index1].data)));

      if (data_equal(&network[target_index1].data, &network[source_index1].data)) {
        switches[switch_index].swap = false;
      } else {
        switches[switch_index].swap = true;
      }
    }
  }

  // second half of the benes network
  for (size_t i = 0; i < level; i++) {
    size_t gap_size = width >> (i + 1);
    size_t group_size = width >> i;
    size_t switches_in_a_group = group_size / 2;

    for (size_t j = 0; j < width; j++) {
      size_t group_id = j >> (level - i);
      size_t in_group_offset = j & (switches_in_a_group - 1);

      size_t target_index1 = (depth - i) * width + j;
      size_t source_index1 = target_index1 - width;
      size_t source_index2 = source_index1 ^ gap_size;
      size_t target_index2 = source_index2 + width;
      size_t switch_index = (depth - i - 1) * width / 2 + group_id * switches_in_a_group + in_group_offset;

      assert((data_equal(&network[target_index1].data,
              &network[source_index1].data) &&
            data_equal(&network[target_index2].data,
              &network[source_index2].data)) ||
          (data_equal(&network[target_index1].data,
                      &network[source_index2].data) &&
           data_equal(&network[target_index2].data,
             &network[source_index1].data)));

      if (data_equal(&network[target_index1].data, &network[source_index1].data)) {
        switches[switch_index].swap = false;
      } else {
        switches[switch_index].swap = true;
      }
    }
  }

  delete[] packets;
  delete[] network;
}

int test_benes_network() {
#define LEVEL 4
#define WIDTH (1 << LEVEL)
#define DEPTH (2 * LEVEL - 1)
  data_t input[WIDTH];
  data_t intermediate[WIDTH * DEPTH - 1];
  data_t output[WIDTH];
  switch_t switches[WIDTH / 2 * DEPTH];
  //srand(time(NULL));

  for (size_t i = 0; i < WIDTH; i++) {
    input[i].addr = rand() % 100;
    input[i].timestamp = rand() % 100;
  }

  do_route(input, intermediate, output, switches, WIDTH, DEPTH);

  for (size_t i = 0; i < WIDTH; i++) {
    printf("%d,%d\t", input[i].addr, input[i].timestamp);
    for (size_t j = 0; j < DEPTH - 1; j++) {
      printf("%d,%d\t", intermediate[j * WIDTH + i].addr, intermediate[j * WIDTH + i].timestamp);
    }
    printf("%d,%d\t", output[i].addr, output[i].timestamp);
    printf("\n");
  }

  for (size_t i = 0; i < WIDTH / 2; i++) {
    for (size_t j = 0; j < DEPTH; j++) {
      if (switches[j * WIDTH / 2 + i].swap) {
        printf("    swap");
      } else {
        printf("    nosw");
      }
    }
    printf("\n");
  }

  return 0;
}
