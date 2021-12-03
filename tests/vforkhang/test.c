// Copyright (c) Open Enclave SDK contributors.
// Licensed under the MIT License.

// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#define _GNU_SOURCE
#include <assert.h>
#include <errno.h>
#include <fcntl.h>
#include <poll.h>
#include <pthread.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/syscall.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>

void siginthandler(int sig)
{
    printf("Received sigint, %i\n", sig);
}

void setup()
{
    signal(SIGINT, siginthandler);
}

void send_sigint()
{
    pid_t fork_pid = vfork();
    if (fork_pid == 0)
    {
        exit(0);
    }
    assert(fork_pid != -1);
}

int vfork_hang()
{
    send_sigint();
}

int main(int argc, const char* argv[])
{
    setup();
    vfork_hang();

    printf("Main: returned from vfork_hang() (%s)\n", argv[0]);
    return 0;
}
