/*
 * file:        unittest-2.c
 * description: libcheck test skeleton, part 2
 */

#define _FILE_OFFSET_BITS 64
#define FUSE_USE_VERSION 26

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <check.h>
#include <zlib.h>
#include <fuse.h>
#include <stdlib.h>
#include <errno.h>


extern struct fuse_operations fs_ops;
extern void block_init(char *file);

/* mockup for fuse_get_context. you can change ctx.uid, ctx.gid in 
 * tests if you want to test setting UIDs in mknod/mkdir
 */
struct fuse_context ctx = { .uid = 500, .gid = 500};
struct fuse_context *fuse_get_context(void)
{
    return &ctx;
}

// /* change test name and make it do something useful */
// START_TEST(a_test)
// {
//     ck_assert_int_eq(1, 1);
// }
// END_TEST

const char *entries[1000];
int entryc;

/* this is an example of a callback function for readdir
 */
int empty_filler(void *ptr, const char *name, const struct stat *stbuf,
                 off_t off)
{
    /* FUSE passes you the entry name and a pointer to a 'struct stat' 
     * with the attributes. Ignore the 'ptr' and 'off' arguments 
     * 
     */
    entries[entryc] = name;
    entryc += 1;
    return 0;
}

void init_entries() {
    for (int i = 0; i < 1000; i++) {
        entries[i] = NULL;
    }
    entryc = 0;
}


struct seen_file{
    char *name;
    int seen;
};

void mark_seen(struct seen_file dir_table[], int count) {
    for (int i = 0; i < count; i++) {
        for (int j = 0; j < count; j++) {
            if (strcmp(entries[i], dir_table[j].name) == 0) {
                dir_table[j].seen ++;
            }
        }
    }
}

void validate_seen(struct seen_file dir_table[], int count) {
    for (int i = 0; i < count; i++) {
        ck_assert_int_eq(1, dir_table[i].seen);
    }    
}

void reset_seen(struct seen_file dir_table[], int count) {
    for (int i = 0; i < count; i++) {
        dir_table[i].seen = 0;
    }    
}



struct seen_file create_root_test[] = {
    {"dir2", 0},
    {"file1", 0},
    {"file2", 0},
    {"file3", 0},
    {"file4", 0},
    {"file5", 0},
    {"file6", 0},
};

struct seen_file create_sub_test[] = {
    {"dir3", 0},
    {"file1", 0},
    {"file2", 0},
    {"file3", 0},
    {"file4", 0},
    {"file5", 0},
    {"file6", 0},
};
struct seen_file create_sub_sub_test[] = {
    {"file1", 0},
    {"file2", 0},
    {"file3", 0},
    {"file4", 0},
    {"file5", 0},
    {"file6", 0},
};


START_TEST(create_test)
{   
    int status;
    int count;

    status = fs_ops.create("/file1", 0100777, NULL);
    ck_assert_int_eq(0, status);

    status = fs_ops.create("/file2", 0100727, NULL);
    ck_assert_int_eq(0, status);

    status = fs_ops.create("/file3", 0100757, NULL);
    ck_assert_int_eq(0, status);

    status = fs_ops.create("/file4", 0100757, NULL);
    ck_assert_int_eq(0, status);

    status = fs_ops.create("/file5", 0100777, NULL);
    ck_assert_int_eq(0, status);

    status = fs_ops.create("/file6", 0100777, NULL);
    ck_assert_int_eq(0, status);

    init_entries();
    fs_ops.readdir("/",  NULL, empty_filler, 0, NULL);
    count = 7;
    ck_assert_int_eq(count, entryc);
    mark_seen(create_root_test, count);
    validate_seen(create_root_test, count);
    reset_seen(create_root_test, count);


    status = fs_ops.create("/dir2/file1", 0100777, NULL);
    ck_assert_int_eq(0, status);

    status = fs_ops.create("/dir2/file2", 0100727, NULL);
    ck_assert_int_eq(0, status);

    status = fs_ops.create("/dir2/file3", 0100757, NULL);
    ck_assert_int_eq(0, status);

    status = fs_ops.create("/dir2/file4", 0100757, NULL);
    ck_assert_int_eq(0, status);

    status = fs_ops.create("/dir2/file5", 0100777, NULL);
    ck_assert_int_eq(0, status);

    status = fs_ops.create("/dir2/file6", 0100777, NULL);
    ck_assert_int_eq(0, status);

    init_entries();
    fs_ops.readdir("/dir2",  NULL, empty_filler, 0, NULL);
    count = 7;
    ck_assert_int_eq(count, entryc);
    mark_seen(create_sub_test, count);
    validate_seen(create_sub_test, count);
    reset_seen(create_sub_test, count);


    status = fs_ops.create("/dir2/dir3/file1", 0100777, NULL);
    ck_assert_int_eq(0, status);

    status = fs_ops.create("/dir2/dir3/file2", 0100727, NULL);
    ck_assert_int_eq(0, status);

    status = fs_ops.create("/dir2/dir3/file3", 0100757, NULL);
    ck_assert_int_eq(0, status);

    status = fs_ops.create("/dir2/dir3/file4", 0100757, NULL);
    ck_assert_int_eq(0, status);

    status = fs_ops.create("/dir2/dir3/file5", 0100777, NULL);
    ck_assert_int_eq(0, status);

    status = fs_ops.create("/dir2/dir3/file6", 0100777, NULL);
    ck_assert_int_eq(0, status);

    init_entries();
    fs_ops.readdir("/dir2/dir3",  NULL, empty_filler, 0, NULL);
    count = 6;
    ck_assert_int_eq(count, entryc);
    mark_seen(create_sub_sub_test, count);
    validate_seen(create_sub_sub_test, count);
    reset_seen(create_sub_sub_test, count);


    // Error testing
    status = fs_ops.create("/file1", 0100777, NULL);
    ck_assert_int_eq(-EEXIST, status);

    status = fs_ops.create("/dir2/file1", 0100777, NULL);
    ck_assert_int_eq(-EEXIST, status);

    status = fs_ops.create("/dir2/dir3/file1", 0100777, NULL);
    ck_assert_int_eq(-EEXIST, status);

    status = fs_ops.create("/dir234/file1", 0100777, NULL);
    ck_assert_int_eq(-ENOENT, status);

    fs_ops.create("/test", 0100777, NULL);

    status = fs_ops.create("/test/file", 0100777, NULL);
    ck_assert_int_eq(-ENOTDIR, status);

    fs_ops.unlink("/test");

    status = fs_ops.create("/dir2", 0100777, NULL);
    ck_assert_int_eq(-EEXIST, status);
}
END_TEST

struct seen_file unlink_root_test[] = {
    {"dir2", 0},
};

struct seen_file unlink_sub_test[] = {
    {"dir3", 0},
};

struct seen_file unlink_sub_sub_test[] = {
};

START_TEST(unlink_test){
    int status;
    int count;

    status = fs_ops.unlink("/file1");
    ck_assert_int_eq(0, status);

    status = fs_ops.unlink("/file2");
    ck_assert_int_eq(0, status);

    status = fs_ops.unlink("/file3");
    ck_assert_int_eq(0, status);

    status = fs_ops.unlink("/file4");
    ck_assert_int_eq(0, status);

    status = fs_ops.unlink("/file5");
    ck_assert_int_eq(0, status);

    status = fs_ops.unlink("/file6");
    ck_assert_int_eq(0, status);

    init_entries();
    fs_ops.readdir("/",  NULL, empty_filler, 0, NULL);
    count = 1;
    ck_assert_int_eq(count, entryc);
    mark_seen(unlink_root_test, count);
    validate_seen(unlink_root_test, count);
    reset_seen(unlink_root_test, count);

    status = fs_ops.unlink("/dir2/file1");
    ck_assert_int_eq(0, status);

    status = fs_ops.unlink("/dir2/file2");
    ck_assert_int_eq(0, status);

    status = fs_ops.unlink("/dir2/file3");
    ck_assert_int_eq(0, status);

    status = fs_ops.unlink("/dir2/file4");
    ck_assert_int_eq(0, status);

    status = fs_ops.unlink("/dir2/file5");
    ck_assert_int_eq(0, status);

    status = fs_ops.unlink("/dir2/file6");
    ck_assert_int_eq(0, status);

    init_entries();
    fs_ops.readdir("/dir2",  NULL, empty_filler, 0, NULL);
    count = 1;
    ck_assert_int_eq(count, entryc);
    mark_seen(unlink_sub_test, count);
    validate_seen(unlink_sub_test, count);
    reset_seen(unlink_sub_test, count);

    status = fs_ops.unlink("/dir2/dir3/file1");
    ck_assert_int_eq(0, status);

    status = fs_ops.unlink("/dir2/dir3/file2");
    ck_assert_int_eq(0, status);

    status = fs_ops.unlink("/dir2/dir3/file3");
    ck_assert_int_eq(0, status);

    status = fs_ops.unlink("/dir2/dir3/file4");
    ck_assert_int_eq(0, status);

    status = fs_ops.unlink("/dir2/dir3/file5");
    ck_assert_int_eq(0, status);

    status = fs_ops.unlink("/dir2/dir3/file6");
    ck_assert_int_eq(0, status);

    init_entries();
    fs_ops.readdir("/dir2/dir3",  NULL, empty_filler, 0, NULL);
    count = 0;
    ck_assert_int_eq(count, entryc);
    mark_seen(unlink_sub_sub_test, count);
    validate_seen(unlink_sub_sub_test, count);
    reset_seen(unlink_sub_sub_test, count);


    //Error testing
    status = fs_ops.unlink("/file1");
    ck_assert_int_eq(-ENOENT, status);
    
    status = fs_ops.unlink("/dir2/file1");
    ck_assert_int_eq(-ENOENT, status);

    status = fs_ops.unlink("/dir2/dir3/file1");
    ck_assert_int_eq(-ENOENT, status);

    status = fs_ops.unlink("/dir2134/file1");
    ck_assert_int_eq(-ENOENT, status);

    fs_ops.create("/test", 0100777, NULL);

    status = fs_ops.unlink("/test/213");
    ck_assert_int_eq(-ENOTDIR, status);

    fs_ops.unlink("/test");

    status = fs_ops.unlink("/dir2");
    ck_assert_int_eq(-EISDIR, status);

}
END_TEST


struct seen_file mkdir_root_test[] = {
    {"dir2", 0},
    {"testdir1", 0},
    {"testdir2", 0},
    {"testdir3", 0},
    {"testdir4", 0},
    {"testdir5", 0},
    {"testdir6", 0},
};

struct seen_file mkdir_sub_test[] = {
    {"dir3", 0},
    {"testdir1", 0},
    {"testdir2", 0},
    {"testdir3", 0},
    {"testdir4", 0},
    {"testdir5", 0},
    {"testdir6", 0},
};

struct seen_file mkdir_sub_sub_test[] = {
    {"testdir1", 0},
    {"testdir2", 0},
    {"testdir3", 0},
    {"testdir4", 0},
    {"testdir5", 0},
    {"testdir6", 0},
};


START_TEST(mkdir_test)
{   
    int status;
    int count;

    status = fs_ops.mkdir("/testdir1", 0777);
    ck_assert_int_eq(0, status);

    status = fs_ops.mkdir("/testdir2", 0727);
    ck_assert_int_eq(0, status);

    status = fs_ops.mkdir("/testdir3", 0757);
    ck_assert_int_eq(0, status);

    status = fs_ops.mkdir("/testdir4", 0757);
    ck_assert_int_eq(0, status);

    status = fs_ops.mkdir("/testdir5", 0777);
    ck_assert_int_eq(0, status);

    status = fs_ops.mkdir("/testdir6", 0777);
    ck_assert_int_eq(0, status);

    init_entries();
    fs_ops.readdir("/",  NULL, empty_filler, 0, NULL);
    count = 7;
    ck_assert_int_eq(count, entryc);
    mark_seen(mkdir_root_test, count);
    validate_seen(mkdir_root_test, count);
    reset_seen(mkdir_root_test, count);


    status = fs_ops.mkdir("/dir2/testdir1", 0777);
    ck_assert_int_eq(0, status);

    status = fs_ops.mkdir("/dir2/testdir2", 0727);
    ck_assert_int_eq(0, status);

    status = fs_ops.mkdir("/dir2/testdir3", 0757);
    ck_assert_int_eq(0, status);

    status = fs_ops.mkdir("/dir2/testdir4", 0757);
    ck_assert_int_eq(0, status);

    status = fs_ops.mkdir("/dir2/testdir5", 0777);
    ck_assert_int_eq(0, status);

    status = fs_ops.mkdir("/dir2/testdir6", 0777);
    ck_assert_int_eq(0, status);

    init_entries();
    fs_ops.readdir("/dir2",  NULL, empty_filler, 0, NULL);
    count = 7;
    ck_assert_int_eq(count, entryc);
    mark_seen(mkdir_sub_test, count);
    validate_seen(mkdir_sub_test, count);
    reset_seen(mkdir_sub_test, count);


    status = fs_ops.mkdir("/dir2/dir3/testdir1", 0777);
    ck_assert_int_eq(0, status);

    status = fs_ops.mkdir("/dir2/dir3/testdir2", 0727);
    ck_assert_int_eq(0, status);

    status = fs_ops.mkdir("/dir2/dir3/testdir3", 0757);
    ck_assert_int_eq(0, status);

    status = fs_ops.mkdir("/dir2/dir3/testdir4", 0757);
    ck_assert_int_eq(0, status);

    status = fs_ops.mkdir("/dir2/dir3/testdir5", 0777);
    ck_assert_int_eq(0, status);

    status = fs_ops.mkdir("/dir2/dir3/testdir6", 0777);
    ck_assert_int_eq(0, status);

    init_entries();
    fs_ops.readdir("/dir2/dir3",  NULL, empty_filler, 0, NULL);
    count = 6;
    ck_assert_int_eq(count, entryc);
    mark_seen(mkdir_sub_sub_test, count);
    validate_seen(mkdir_sub_sub_test, count);
    reset_seen(mkdir_sub_sub_test, count);


    // Error testing
    status = fs_ops.mkdir("/testdir1", 0777);
    ck_assert_int_eq(-EEXIST, status);

    status = fs_ops.mkdir("/dir2/testdir1", 0777);
    ck_assert_int_eq(-EEXIST, status);

    status = fs_ops.mkdir("/dir2/dir3/testdir1", 0777);
    ck_assert_int_eq(-EEXIST, status);

    status = fs_ops.mkdir("/dir234/testdir1", 0777);
    ck_assert_int_eq(-ENOENT, status);

    fs_ops.create("/test", 0100777, NULL);

    status = fs_ops.mkdir("/test/testdir", 0777);
    ck_assert_int_eq(-ENOTDIR, status);

    fs_ops.unlink("/test");

    status = fs_ops.mkdir("/dir2", 0777);
    ck_assert_int_eq(-EEXIST, status);
}
END_TEST


struct seen_file rmdir_root_test[] = {
    {"dir2", 0},
};

struct seen_file rmdir_sub_test[] = {
    {"dir3", 0},
};

struct seen_file rmdir_sub_sub_test[] = {
};

START_TEST(rmdir_test){
    int status;
    int count;

    status = fs_ops.rmdir("/testdir1");
    ck_assert_int_eq(0, status);

    status = fs_ops.rmdir("/testdir2");
    ck_assert_int_eq(0, status);

    status = fs_ops.rmdir("/testdir3");
    ck_assert_int_eq(0, status);

    status = fs_ops.rmdir("/testdir4");
    ck_assert_int_eq(0, status);

    status = fs_ops.rmdir("/testdir5");
    ck_assert_int_eq(0, status);

    status = fs_ops.rmdir("/testdir6");
    ck_assert_int_eq(0, status);

    init_entries();
    fs_ops.readdir("/",  NULL, empty_filler, 0, NULL);
    count = 1;
    ck_assert_int_eq(count, entryc);
    mark_seen(rmdir_root_test, count);
    validate_seen(rmdir_root_test, count);
    reset_seen(rmdir_root_test, count);

    status = fs_ops.rmdir("/dir2/testdir1");
    ck_assert_int_eq(0, status);

    status = fs_ops.rmdir("/dir2/testdir2");
    ck_assert_int_eq(0, status);

    status = fs_ops.rmdir("/dir2/testdir3");
    ck_assert_int_eq(0, status);

    status = fs_ops.rmdir("/dir2/testdir4");
    ck_assert_int_eq(0, status);

    status = fs_ops.rmdir("/dir2/testdir5");
    ck_assert_int_eq(0, status);

    status = fs_ops.rmdir("/dir2/testdir6");
    ck_assert_int_eq(0, status);

    init_entries();
    fs_ops.readdir("/dir2",  NULL, empty_filler, 0, NULL);
    count = 1;
    ck_assert_int_eq(count, entryc);
    mark_seen(rmdir_sub_test, count);
    validate_seen(rmdir_sub_test, count);
    reset_seen(rmdir_sub_test, count);

    status = fs_ops.rmdir("/dir2/dir3/testdir1");
    ck_assert_int_eq(0, status);

    status = fs_ops.rmdir("/dir2/dir3/testdir2");
    ck_assert_int_eq(0, status);

    status = fs_ops.rmdir("/dir2/dir3/testdir3");
    ck_assert_int_eq(0, status);

    status = fs_ops.rmdir("/dir2/dir3/testdir4");
    ck_assert_int_eq(0, status);

    status = fs_ops.rmdir("/dir2/dir3/testdir5");
    ck_assert_int_eq(0, status);

    status = fs_ops.rmdir("/dir2/dir3/testdir6");
    ck_assert_int_eq(0, status);

    init_entries();
    fs_ops.readdir("/dir2/dir3",  NULL, empty_filler, 0, NULL);
    count = 0;
    ck_assert_int_eq(count, entryc);
    mark_seen(rmdir_sub_sub_test, count);
    validate_seen(rmdir_sub_sub_test, count);
    reset_seen(rmdir_sub_sub_test, count);


    //Error testing
    status = fs_ops.rmdir("/testdir1");
    ck_assert_int_eq(-ENOENT, status);
    
    status = fs_ops.rmdir("/dir2/testdir1");
    ck_assert_int_eq(-ENOENT, status);

    status = fs_ops.rmdir("/dir2/dir3/testdir1");
    ck_assert_int_eq(-ENOENT, status);

    status = fs_ops.rmdir("/dir2134/testdir1");
    ck_assert_int_eq(-ENOENT, status);

    fs_ops.create("/test", 0100777, NULL);

    status = fs_ops.rmdir("/test/213");
    ck_assert_int_eq(-ENOTDIR, status);

    status = fs_ops.rmdir("/test");
    ck_assert_int_eq(-ENOTDIR, status);

    fs_ops.unlink("/test");

    status = fs_ops.rmdir("/dir2");
    ck_assert_int_eq(-ENOTEMPTY, status);

}
END_TEST

int read_file_by_n(const char *path, char *buf, int n) {
    int len = 0;
    int len_to_read = n;
    int len_read = fs_ops.read(path, buf+len, len_to_read, len, NULL);
    while (len_read != 0) {
        if (len_read < len_to_read) {
            len_to_read = len_read;
        } else {
            len += len_read;
        }
        len_read = fs_ops.read(path, buf+len, len_to_read, len, NULL);
    }
    return len;
}



START_TEST(write_test){

    char readbuf[12300]; // make this bigger than 3 blocks
    char *ptr, *buf = malloc(4010);
    int i;
    for (i=0, ptr = buf; ptr < buf+4000; i++){
        ptr += sprintf(ptr, "%d ", i);
    }

    int pf_blocks;
    struct statvfs st;
    fs_ops.statfs(NULL, &st);

    pf_blocks = st.f_bfree;

    fs_ops.create("test.4k-", 0100777, NULL);
    int status;
    status = fs_ops.write("test.4k-", buf, 4010, 0, NULL);
    ck_assert_int_eq(4010, status);
    read_file_by_n("test.4k-", readbuf, 4010);
    status = memcmp(readbuf,buf, 4010);
    ck_assert_int_eq(0, status);
    
    fs_ops.statfs(NULL, &st);
    ck_assert_int_eq(pf_blocks-2, st.f_bfree);
    pf_blocks = st.f_bfree;

    
    char *ptr2, *buf2 = malloc(4096);
    for (i=0, ptr2 = buf2; ptr2 < buf2+4000; i++){
        ptr2 += sprintf(ptr2, "%d ", i);
    }

    fs_ops.create("test.4k", 0100777, NULL);
    status = fs_ops.write("test.4k", buf2, 4096, 0, NULL);
    ck_assert_int_eq(4096, status);

    read_file_by_n("test.4k", readbuf, 4096);
    status = memcmp(readbuf,buf2, 4096);
    ck_assert_int_eq(0, status);

    fs_ops.statfs(NULL, &st);
    ck_assert_int_eq(pf_blocks-2, st.f_bfree);
    pf_blocks = st.f_bfree;
        
    char *ptr3, *buf3 = malloc(8096);
    for (i=0, ptr3 = buf3; ptr3 < buf3+4000; i++){
        ptr3 += sprintf(ptr3, "%d ", i);
    }

    fs_ops.create("test.8k-", 0100777, NULL);
    status = fs_ops.write("test.8k-", buf3, 8096, 0, NULL);
    ck_assert_int_eq(8096, status);


    read_file_by_n("test.8k-", readbuf, 8096);
    status = memcmp(readbuf,buf3, 8096);
    ck_assert_int_eq(0, status);

    fs_ops.statfs(NULL, &st);
    ck_assert_int_eq(pf_blocks-3, st.f_bfree);
    pf_blocks = st.f_bfree;
    

    char *ptr4, *buf4 = malloc(8192);
    for (i=0, ptr4 = buf4; ptr4 < buf4+4000; i++){
        ptr4 += sprintf(ptr4, "%d ", i);
    }

    fs_ops.create("test.8k", 0100777, NULL);
    status = fs_ops.write("test.8k", buf4, 8192, 0, NULL);
    ck_assert_int_eq(8192, status);

    read_file_by_n("test.8k", readbuf, 8192);
    status = memcmp(readbuf, buf4, 8192);
    ck_assert_int_eq(0, status);

    fs_ops.statfs(NULL, &st);
    ck_assert_int_eq(pf_blocks-3, st.f_bfree);
    pf_blocks = st.f_bfree;

    char *ptr5, *buf5 = malloc(12192);
    for (i=0, ptr5 = buf5; ptr5 < buf5+4000; i++){
        ptr5 += sprintf(ptr5, "%d ", i);
    }

    fs_ops.create("test.12k-", 0100777, NULL);
    status = fs_ops.write("test.12k-", buf5, 12192, 0, NULL);
    ck_assert_int_eq(12192, status);


    read_file_by_n("test.12k-", readbuf, 12192);
    status = memcmp(readbuf, buf5, 12192);
    ck_assert_int_eq(0, status);

    fs_ops.statfs(NULL, &st);
    ck_assert_int_eq(pf_blocks-4, st.f_bfree);
    pf_blocks = st.f_bfree;

    char *ptr6, *buf6 = malloc(12288);
    for (i=0, ptr6 = buf6; ptr6 < buf6+4000; i++){
        ptr6 += sprintf(ptr6, "%d ", i);
    }

    fs_ops.create("test.12k", 0100777, NULL);
    status = fs_ops.write("test.12k", buf6, 12288, 0, NULL);
    ck_assert_int_eq(12288, status);


    read_file_by_n("test.12k", readbuf, 12288);
    status = memcmp(readbuf, buf6, 12288);
    ck_assert_int_eq(0, status);

    fs_ops.statfs(NULL, &st);
    ck_assert_int_eq(pf_blocks-4, st.f_bfree);
    pf_blocks = st.f_bfree;

    for (i=0, ptr6 = buf6; ptr6 < buf6+4000; i++){
        ptr6 += sprintf(ptr6, "%d ", i+23);
    }

    status = fs_ops.write("test.12k", buf6, 12288, 0, NULL);
    read_file_by_n("test.12k", readbuf, 12288);
    status = memcmp(readbuf, buf6, 12288);
    ck_assert_int_eq(0, status);

    fs_ops.statfs(NULL, &st);
    ck_assert_int_eq(pf_blocks, st.f_bfree);
    pf_blocks = st.f_bfree;
    

    // ERROR checking

    status = fs_ops.write("error", buf6, 12288, 0, NULL);
    ck_assert_int_eq(-ENOENT, status);

    status = fs_ops.write("sdsa/error", buf6, 12288, 0, NULL);
    ck_assert_int_eq(-ENOENT, status);

    fs_ops.create("/test", 0100777, NULL);

    status = fs_ops.write("test/error", buf6, 12288, 0, NULL);
    ck_assert_int_eq(-ENOTDIR, status);

    status = fs_ops.write("test", buf6, 12288, 123, NULL);
    ck_assert_int_eq(-EINVAL, status);

    fs_ops.unlink("/test");

}
END_TEST


START_TEST(truncate_test){
    int pf_blocks;
    int status;
    struct statvfs st;
    fs_ops.statfs(NULL, &st);
    pf_blocks = st.f_bfree;

    status = fs_ops.truncate("test.4k-", 0);
    ck_assert_int_eq(0, status);

    fs_ops.statfs(NULL, &st);
    ck_assert_int_eq(pf_blocks+1, st.f_bfree);
    pf_blocks = st.f_bfree;

    status = fs_ops.truncate("test.4k", 0);
    ck_assert_int_eq(0, status);

    fs_ops.statfs(NULL, &st);
    ck_assert_int_eq(pf_blocks+1, st.f_bfree);
    pf_blocks = st.f_bfree;

    status = fs_ops.truncate("test.8k-", 0);
    ck_assert_int_eq(0, status);

    fs_ops.statfs(NULL, &st);
    ck_assert_int_eq(pf_blocks+2, st.f_bfree);
    pf_blocks = st.f_bfree;

    status = fs_ops.truncate("test.8k", 0);
    ck_assert_int_eq(0, status);

    fs_ops.statfs(NULL, &st);
    ck_assert_int_eq(pf_blocks+2, st.f_bfree);
    pf_blocks = st.f_bfree;

    status = fs_ops.truncate("test.12k-", 0);
    ck_assert_int_eq(0, status);

    fs_ops.statfs(NULL, &st);
    ck_assert_int_eq(pf_blocks+3, st.f_bfree);
    pf_blocks = st.f_bfree;

    status = fs_ops.truncate("test.12k", 0);
    ck_assert_int_eq(0, status);

    fs_ops.statfs(NULL, &st);
    ck_assert_int_eq(pf_blocks + 3, st.f_bfree);
    pf_blocks = st.f_bfree;

    //error checking

    status = fs_ops.truncate("test.12k", 123);
    ck_assert_int_eq(-EINVAL, status);

    status = fs_ops.truncate("dir2", 0);
    ck_assert_int_eq(-EISDIR, status);

    status = fs_ops.truncate("/asdr32/asd213", 0);
    ck_assert_int_eq(-ENOENT, status);

    fs_ops.create("/test", 0100777, NULL);

    status = fs_ops.truncate("test/error", 0);
    ck_assert_int_eq(-ENOTDIR, status);

    fs_ops.unlink("test");

}
END_TEST


START_TEST(utime_test){
    struct utimbuf ut;
    ut.modtime = time(NULL);
    int status;
    struct stat sb;
    status = fs_ops.utime("/test.12k", &ut);
    ck_assert_int_eq(0, status);
    fs_ops.getattr("/test.12k", &sb);
    ck_assert_int_eq(ut.modtime, sb.st_mtime);

    status = fs_ops.utime("dir2", &ut);
    ck_assert_int_eq(0, status);
    fs_ops.getattr("dir2", &sb);
    ck_assert_int_eq(ut.modtime, sb.st_mtime);

    //Error Checking

    status = fs_ops.utime("/asdr32/asd213", &ut);
    ck_assert_int_eq(-ENOENT, status);

    fs_ops.create("/test", 0100777, NULL);

    status = fs_ops.utime("test/error", &ut);
    ck_assert_int_eq(-ENOTDIR, status);

    fs_ops.unlink("test");

}
END_TEST

int main(int argc, char **argv)
{
    block_init("test2.img");
    fs_ops.init(NULL);
    fs_ops.mkdir("/dir2", 0777);
    fs_ops.mkdir("/dir2/dir3", 0777);
    
    Suite *s = suite_create("fs5600");
    TCase *tc = tcase_create("write_mostly");

    tcase_add_test(tc, create_test);
    tcase_add_test(tc, unlink_test);
    tcase_add_test(tc, mkdir_test);
    tcase_add_test(tc, rmdir_test);
    tcase_add_test(tc, write_test);
    tcase_add_test(tc, truncate_test);
    tcase_add_test(tc, utime_test);

    suite_add_tcase(s, tc);
    SRunner *sr = srunner_create(s);
    srunner_set_fork_status(sr, CK_NOFORK);
    
    srunner_run_all(sr, CK_VERBOSE);
    int n_failed = srunner_ntests_failed(sr);
    printf("%d tests failed\n", n_failed);
    
    srunner_free(sr);
    return (n_failed == 0) ? EXIT_SUCCESS : EXIT_FAILURE;
}

