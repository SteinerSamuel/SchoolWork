/*
 * file:        part-1.c
 * description: Part 1, CS5600 Project 1, Spring 2021
 */

/* THE ONLY INCLUDE FILE */
#include "sysdefs.h"

#define STDIN_FD 0
#define STDOUT_FD 1
#define MAX_LINE_SIZE 200
#define PROMT_SIGN ">"
#define WELCOME_MSG "Hello, type lines of input, or 'quit':"

/* write these functions */

int read(int fd, void *ptr, int len);
int write(int fd, void *ptr, int len);
void exit(int err);

/* Read a line */
void readline(char *buf, int len);

/* Print without new line */
void print(char *buf);

/* Print with a new line */
void println(char *buf);

/* ---------- */

/* Factor, factor! Don't put all your code in main()! 
*/

/* read one line from stdin (file descriptor 0) into a buffer: */

/* print a string to stdout (file descriptor 1) */

/* ---------- */

void main(void)
{
	/* your code here */
    println(WELCOME_MSG);
    while(1) {
        print(PROMT_SIGN);
        char msg[MAX_LINE_SIZE];
        readline(msg, MAX_LINE_SIZE);
        if (msg[0] == 'q' && msg[1] == 'u' && msg[2] == 'i' && msg[3] == 't') {
            exit(0);
        }
        println(msg);
    }
}

int read(int fd, void *ptr, int len) {
    syscall(__NR_read, fd, ptr, len);
    return 0;
}

int write(int fd, void *ptr, int len) {
    syscall(__NR_write, fd, ptr, len);
    return 0;
}

void exit(int err) {
    syscall(__NR_exit, err);
}

void readline(char *buf, int len) {
    int i = 0;
    // Read one char/byte at a time.
    char currChar[1];
    while(i < len) {
       read(STDIN_FD, currChar, sizeof(char)); 
       if (currChar[0] == '\n') {
           buf[i] = 0;
           break;
       }
       buf[i] = currChar[0];
       i += 1;
    }
}

void print(char *buf) {
    int i = 0;
    while (buf[i] != 0) {
        write(STDOUT_FD, buf + i, sizeof(char));
        i += 1;
    }
}

void println(char *buf) {
    print(buf);
    char newline[] = "\n";
    write(STDOUT_FD, newline, sizeof(newline));
}

