#include <execinfo.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <common/memory.h>

void handler(int sig, siginfo_t *si, void* unused) {
  void *array[10];
  size_t size;

  // get void*'s for all entries on the stack
  size = backtrace(array, 10);

  // print out all the frames to stderr
  fprintf(stderr, "Error: signal %d:\n", sig);
  backtrace_symbols_fd(array, size, STDERR_FILENO);

  fprintf(stderr, "Peak memory usage: %ld\n", getPeakRSS());
  fprintf(stderr, "Current memory usage: %ld\n", getCurrentRSS());
  exit(1);
}

void register_handler() {
  struct sigaction sa;
  sa.sa_flags = SA_SIGINFO;
  sigemptyset(&sa.sa_mask);
  sa.sa_sigaction = &handler;
  if (sigaction(SIGSEGV, &sa, NULL) == -1) {
    fprintf(stderr, "Failed to install handler for SIGSEGV.\n");
  }
  if (sigaction(SIGBUS, &sa, NULL) == -1) {
    fprintf(stderr, "Failed to install handler for SIGBUS.\n");
  }
}
