#define _GNU_SOURCE

#include <stdint.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>

/* Freeze wall-clock seed sources while leaving monotonic timing untouched. */
static time_t replay_epoch(void) {
    const char *raw = getenv("REVENGEBENCH_REPLAY_EPOCH");
    if (raw == NULL || *raw == '\0') {
        return (time_t)20260813;
    }
    return (time_t)strtoll(raw, NULL, 10);
}

time_t time(time_t *result) {
    time_t value = replay_epoch();
    if (result != NULL) {
        *result = value;
    }
    return value;
}

int gettimeofday(struct timeval *tv, void *tz) {
    (void)tz;
    if (tv != NULL) {
        tv->tv_sec = replay_epoch();
        tv->tv_usec = 0;
    }
    return 0;
}
