// shim_syscall_cp.c
#define _GNU_SOURCE
#include <stdarg.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>

long __syscall_cp(long n, ...) {
    va_list ap;
    long a[6] = {0};
    va_start(ap, n);
    for (int i = 0; i < 6; i++) a[i] = va_arg(ap, long);
    va_end(ap);

    if (n == SYS_ppoll) {
        // a[0]=fds, a[1]=nfds, a[2]=const struct timespec *tmo, a[3]=sigmask, a[4]=sigsetsize
        const struct timespec *tmo = (const struct timespec *)a[2];
        if (!tmo) return 0;
        struct timespec ts = *tmo;
        return nanosleep(&ts, NULL);
    }
    // 其他系统调用：透传
    return syscall(n, a[0], a[1], a[2], a[3], a[4], a[5]);
}

