/*
 * file:        part-2.c
 * description: Part 2, CS5600 Project 1, Spring 2021
 */

/* NO OTHER INCLUDE FILES */
#include "elf64.h"
#include "sysdefs.h"

#define STDIN_FD 0
#define STDOUT_FD 1
#define PROMT_SIGN ">"
#define MAX_LINE_SIZE 200
#define MAX_ARGV_NUM 10

extern void *vector[];

/* ---------- */

/* write these functions 
*/
int read(int fd, void *ptr, int len);
int write(int fd, void *ptr, int len);
void exit(int err);
int open(char *path, int flags);
int close(int fd);
int lseek(int fd, int offset, int flag);
void *mmap(void *addr, int len, int prot, int flags, int fd, int offset);
int munmap(void *addr, int len);
int exec_file(char *filename);

/* ---------- */

/* the three 'system call' functions - readline, print, getarg 
 * hints: 
 *  - read() or write() one byte at a time. It's OK to be slow.
 *  - stdin is file desc. 0, stdout is file descriptor 1
 *  - use global variables for getarg
 */

void do_readline(char *buf, int len);
void do_print(char *buf);
char *do_getarg(int i);         

/* ---------- */

/* the guts of part 2
 *   read the ELF header
 *   for each section, if b_type == PT_LOAD:
 *     create mmap region
 *     read from file into region
 *   function call to hdr.e_entry
 *   munmap each mmap'ed region so we don't crash the 2nd time
 */

/* your code here */

/* ---------- */

/* simple function to split a line:
 *   char buffer[200];
 *   <read line into 'buffer'>
 *   char *argv[10];
 *   int argc = split(argv, 10, buffer);
 *   ... pointers to words are in argv[0], ... argv[argc-1]
 */
int split(char **argv, int max_argc, char *line)
{
	int i = 0;
	char *p = line;

	while (i < max_argc) {
		while (*p != 0 && (*p == ' ' || *p == '\t' || *p == '\n'))
			*p++ = 0;
		if (*p == 0)
			return i;
		argv[i++] = p;
		while (*p != 0 && *p != ' ' && *p != '\t' && *p != '\n')
			p++;
	}
	return i;
}

/* ---------- */
int argc;
char *argv[MAX_ARGV_NUM];

void main(void)
{
	vector[0] = do_readline;
	vector[1] = do_print;
	vector[2] = do_getarg;

	/* YOUR CODE HERE */
    while (1) {
        do_print(PROMT_SIGN);
        char line[MAX_LINE_SIZE]; 
        do_readline(line, MAX_LINE_SIZE);
        if (line[0] == 'q' && line[1] == 'u' && line[2] == 'i' && line[3] == 't') {
            exit(0);
        }
        argc = split(argv, MAX_ARGV_NUM, line);
        if (argc == 0) {
            do_print("Insufficient arguments.\n");
            continue;
        }
        if (exec_file(argv[0])) {
            do_print("Program doesn't exist or something else is wrong.i\n");
        };
    }
    exec_file("wait");
    exec_file("hello"); 
    do_print("something");
    exit(0);
}

int exec_file(char *filename) {
    int fd;
    long offset = 0x80000000;
    if ((fd = open(filename, O_RDONLY)) < 0) {
        return 1;
    }
    /* read the main header (offset 0) */
    struct elf64_ehdr hdr;
    read(fd, &hdr, sizeof(hdr));

    /* read program headers (offset 'hdr.e_phoff') */
    int n = hdr.e_phnum;
    struct elf64_phdr phdrs[n];
    lseek(fd, hdr.e_phoff, SEEK_SET);
    read(fd, phdrs, sizeof(phdrs));

    
    /* look at each section in program headers */
    // Make sure the address passed in to mmap also is a multiple of 4096
    void *addr = (void *)offset;
    int numOfLoad = 4;
    void *addrArr[4];
    int lenArr[4];
    int j = 0;  // index to keep track of the address and lenth array 
    for (int i = 0; i < hdr.e_phnum; i++) {
        if (phdrs[i].p_type == PT_LOAD) {
            int len = ROUND_UP(phdrs[i].p_memsz, 4096);
            addrArr[j] = addr;
            lenArr[j] = len;
            void *buf = mmap(addr, len, PROT_READ | PROT_WRITE |
                      PROT_EXEC, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
            if (buf == MAP_FAILED) {
                do_print("mmap failed\n");
                exit(1);
            }
            lseek(fd, (int)phdrs[i].p_offset, SEEK_SET);
            read(fd, buf, (int)phdrs[i].p_filesz);
            addr += len;
            j += 1;
        }
    }

    void (*f)() = hdr.e_entry + offset;
    f();
    close(fd);

    for (int i = 0; i < numOfLoad; i++) {
        munmap(addrArr[i], lenArr[i]);
    }

    return 0;
}

int read(int fd, void *ptr, int len) {
    return syscall(__NR_read, fd, ptr, len);
}

int write(int fd, void *ptr, int len) {
    return syscall(__NR_write, fd, ptr, len);
}

void exit(int err) {
    syscall(__NR_exit, err);
}

void do_readline(char *buf, int len) {
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

void do_print(char *buf) {
    int i = 0;
    while (buf[i] != 0) {
        write(STDOUT_FD, buf + i, sizeof(char));
        i += 1;
    }
}

char *do_getarg(int i) {
    if (i < argc) {
        return argv[i];
    } else {
        return 0;
    }
}

int open(char *path, int flags) {
    return syscall(__NR_open, path, flags);
}

int close(int fd) {
    return syscall(__NR_close, fd);
}

int lseek(int fd, int offset, int flag) {
    return syscall(__NR_lseek, fd, offset, flag);
}

void *mmap(void *addr, int len, int prot, int flags, int fd, int offset) {
    return (void *)syscall(__NR_mmap, addr, len, prot, flags, fd, offset);
}

int munmap(void *addr, int len) {
    return syscall(__NR_munmap, addr, len);
}

