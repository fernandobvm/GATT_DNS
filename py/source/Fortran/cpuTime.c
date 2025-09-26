#include <sys/resource.h>

double get_cpu_time() {
    struct rusage usage;
    getrusage(RUSAGE_SELF, &usage);
    return (double)usage.ru_utime.tv_sec + (double)usage.ru_utime.tv_usec / 1e6 +
           (double)usage.ru_stime.tv_sec + (double)usage.ru_stime.tv_usec / 1e6;
}

// gcc -fPIC -shared -o libcpu_usage.so cpu_usage.c