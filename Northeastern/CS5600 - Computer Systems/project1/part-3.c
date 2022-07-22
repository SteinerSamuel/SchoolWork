/*
 * file:        part-3.c
 * description: part 3, CS5600 Project 1, Spring 2021
 */

/* NO OTHER INCLUDE FILES */
#include "elf64.h"
#include "sysdefs.h"

extern void *vector[];
extern void switch_to(void **location_for_old_sp, void *new_value);
extern void *setup_stack0(void *_stack, void *func);

/* ---------- */

#define STDIN_FD 0
#define STDOUT_FD 1

void * p1;
void * p2;
void * p0;

/* write these 
*/
int read(int fd, void *ptr, int len);
int write(int fd, void *ptr, int len);
void exit(int err);
int open(char *path, int flags);
int close(int fd);
int lseek(int fd, int offset, int flag);
void *mmap(void *addr, int len, int prot, int flags, int fd, int offset);
int munmap(void *addr, int len);

/* ---------- */

/* copy from Part 2 */
void do_print(char *buf);
void * load_file_w_offset(char * file, long offset);

/* ---------- */

/* write these new functions */
void do_yield12(void);
void do_yield21(void);
void do_uexit(void);

/* ---------- */

void main(void)
{
	vector[1] = do_print;

	vector[3] = do_yield12;
	vector[4] = do_yield21;
	vector[5] = do_uexit;

	/* your code here */

    //  Set up both stacks
	char stack1[4096];
    char stack2[4096];

    //  load process 1
    void * process1p = load_file_w_offset("process1", 0x80000000);
    void (*f)() = process1p;

    // load process 2
    void * process2p = load_file_w_offset("process2", 0x90000000);
    void (*g)() = process2p;

    // setup the stacks for process 1 and 2
    p1 = setup_stack0(stack1+4096, f);
    p2 = setup_stack0(stack2+4096, g);

    // switch to process 1 and let them run
    switch_to(&p0, p1);

    // done
	do_print("done\n");
	exit(0);
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

void do_print(char *buf) {
    int i = 0;
    while (buf[i] != 0) {
        write(STDOUT_FD, buf + i, sizeof(char));
        i += 1;
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


void * load_file_w_offset(char *filename, long offset) {
    int fd;
    if ((fd = open(filename, O_RDONLY)) < 0) {
        exit(1);
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
    // void *addr = (void *)offset;
	void *addr = (void *)offset;
    int j = 0;  // index to keep track of the address and lenth array 
    for (int i = 0; i < hdr.e_phnum; i++) {
        if (phdrs[i].p_type == PT_LOAD) {
            int len = ROUND_UP(phdrs[i].p_memsz, 4096);
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
	return (hdr.e_entry + offset);
}


void do_yield12(void) {
    switch_to(&p1, p2);
    return;
}

void do_yield21(void) {
    switch_to(&p2, p1);
    return ;
}

void do_uexit(void) {
    switch_to(&p1, p0);
    return;
}