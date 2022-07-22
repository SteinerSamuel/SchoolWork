/*
 * file:        testing.c
 * description: libcheck test skeleton for file system project
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

/* note that your tests will call:
 *  fs_ops.getattr(path, struct stat *sb)
 *  fs_ops.readdir(path, NULL, filler_function, 0, NULL)
 *  fs_ops.read(path, buf, len, offset, NULL);
 *  fs_ops.statfs(path, struct statvfs *sv);
 */
extern struct fuse_operations fs_ops;
extern void block_init(char *file);

struct seen_file{
    char *name;
    int seen;
};

struct seen_file dir1_table[] = {
    {"dir2", 0},
    {"dir3", 0},
    {"dir-with-long-name", 0},
    {"file.10", 0},
    {"file.1k", 0},
    {"file.8k+", 0},
};

struct seen_file dir2_table[] = {
    {"twenty-seven-byte-file-name", 0},
    {"file.4k+", 0},
};

struct seen_file dir3_table[] = {
    {"subdir", 0},
    {"file.12k-", 0},
};

struct seen_file dir4_table[] = {
    {"file.4k-", 0},
    {"file.8k-", 0},
    {"file.12k", 0},
};

struct seen_file dir5_table[] = {
    {"file.12k+", 0},
};

START_TEST(test_getattr)
{
    int status;
    struct stat sb;
    status = fs_ops.getattr("/", &sb);
    ck_assert_int_eq(0, status);
    ck_assert_int_eq(0, sb.st_uid);
    ck_assert_int_eq(0, sb.st_gid);
    ck_assert_int_eq(040777, sb.st_mode);
    ck_assert_int_eq(4096, sb.st_size);
    ck_assert_int_eq(1565283152, sb.st_ctime);
    ck_assert_int_eq(1565283167, sb.st_mtime);

    status = fs_ops.getattr("/file.1k", &sb);
    ck_assert_int_eq(0, status);
    ck_assert_int_eq(500, sb.st_uid);
    ck_assert_int_eq(500, sb.st_gid);
    ck_assert_int_eq(0100666, sb.st_mode);
    ck_assert_int_eq(1000, sb.st_size);
    ck_assert_int_eq(1565283152, sb.st_ctime);
    ck_assert_int_eq(1565283152, sb.st_mtime);

    status = fs_ops.getattr("/file.10", &sb);
    ck_assert_int_eq(0, status);
    ck_assert_int_eq(500, sb.st_uid);
    ck_assert_int_eq(500, sb.st_gid);
    ck_assert_int_eq(0100666, sb.st_mode);
    ck_assert_int_eq(10, sb.st_size);
    ck_assert_int_eq(1565283152, sb.st_ctime);
    ck_assert_int_eq(1565283167, sb.st_mtime);

    status = fs_ops.getattr("/dir-with-long-name", &sb);
    ck_assert_int_eq(0, status);
    ck_assert_int_eq(0, sb.st_uid);
    ck_assert_int_eq(0, sb.st_gid);
    ck_assert_int_eq(040777, sb.st_mode);
    ck_assert_int_eq(4096, sb.st_size);
    ck_assert_int_eq(1565283152, sb.st_ctime);
    ck_assert_int_eq(1565283167, sb.st_mtime);

    status = fs_ops.getattr("/dir-with-long-name/file.12k+", &sb);
    ck_assert_int_eq(0, status);
    ck_assert_int_eq(0, sb.st_uid);
    ck_assert_int_eq(500, sb.st_gid);
    ck_assert_int_eq(0100666, sb.st_mode);
    ck_assert_int_eq(12289, sb.st_size);
    ck_assert_int_eq(1565283152, sb.st_ctime);
    ck_assert_int_eq(1565283167, sb.st_mtime);

    status = fs_ops.getattr("/dir2", &sb);
    ck_assert_int_eq(0, status);
    ck_assert_int_eq(500, sb.st_uid);
    ck_assert_int_eq(500, sb.st_gid);
    ck_assert_int_eq(040777, sb.st_mode);
    ck_assert_int_eq(8192, sb.st_size);
    ck_assert_int_eq(1565283152, sb.st_ctime);
    ck_assert_int_eq(1565283167, sb.st_mtime);

    status = fs_ops.getattr("/dir2/twenty-seven-byte-file-name", &sb);
    ck_assert_int_eq(0, status);
    ck_assert_int_eq(500, sb.st_uid);
    ck_assert_int_eq(500, sb.st_gid);
    ck_assert_int_eq(0100666, sb.st_mode);
    ck_assert_int_eq(1000, sb.st_size);
    ck_assert_int_eq(1565283152, sb.st_ctime);
    ck_assert_int_eq(1565283167, sb.st_mtime);

    status = fs_ops.getattr("/dir2/file.4k+", &sb);
    ck_assert_int_eq(0, status);
    ck_assert_int_eq(500, sb.st_uid);
    ck_assert_int_eq(500, sb.st_gid);
    ck_assert_int_eq(0100777, sb.st_mode);
    ck_assert_int_eq(4098, sb.st_size);
    ck_assert_int_eq(1565283152, sb.st_ctime);
    ck_assert_int_eq(1565283167, sb.st_mtime);

    status = fs_ops.getattr("/dir3", &sb);
    ck_assert_int_eq(0, status);
    ck_assert_int_eq(0, sb.st_uid);
    ck_assert_int_eq(500, sb.st_gid);
    ck_assert_int_eq(040777, sb.st_mode);
    ck_assert_int_eq(4096, sb.st_size);
    ck_assert_int_eq(1565283152, sb.st_ctime);
    ck_assert_int_eq(1565283167, sb.st_mtime);

    fs_ops.getattr("/dir3/subdir", &sb);
    ck_assert_int_eq(0, status);
    ck_assert_int_eq(0, sb.st_uid);
    ck_assert_int_eq(500, sb.st_gid);
    ck_assert_int_eq(040777, sb.st_mode);
    ck_assert_int_eq(4096, sb.st_size);
    ck_assert_int_eq(1565283152, sb.st_ctime);
    ck_assert_int_eq(1565283167, sb.st_mtime);

    status = fs_ops.getattr("/dir3/subdir/file.4k-", &sb);
    ck_assert_int_eq(0, status);
    ck_assert_int_eq(500, sb.st_uid);
    ck_assert_int_eq(500, sb.st_gid);
    ck_assert_int_eq(0100666, sb.st_mode);
    ck_assert_int_eq(4095, sb.st_size);
    ck_assert_int_eq(1565283152, sb.st_ctime);
    ck_assert_int_eq(1565283167, sb.st_mtime);

    status = fs_ops.getattr("/dir3/subdir/file.8k-", &sb);
    ck_assert_int_eq(0, status);
    ck_assert_int_eq(500, sb.st_uid);
    ck_assert_int_eq(500, sb.st_gid);
    ck_assert_int_eq(0100666, sb.st_mode);
    ck_assert_int_eq(8190, sb.st_size);
    ck_assert_int_eq(1565283152, sb.st_ctime);
    ck_assert_int_eq(1565283167, sb.st_mtime);

    status = fs_ops.getattr("/dir3/subdir/file.12k", &sb);
    ck_assert_int_eq(0, status);
    ck_assert_int_eq(500, sb.st_uid);
    ck_assert_int_eq(500, sb.st_gid);
    ck_assert_int_eq(0100666, sb.st_mode);
    ck_assert_int_eq(12288, sb.st_size);
    ck_assert_int_eq(1565283152, sb.st_ctime);
    ck_assert_int_eq(1565283167, sb.st_mtime);

    status = fs_ops.getattr("/dir3/file.12k-", &sb);
    ck_assert_int_eq(0, status);
    ck_assert_int_eq(0, sb.st_uid);
    ck_assert_int_eq(500, sb.st_gid);
    ck_assert_int_eq(0100777, sb.st_mode);
    ck_assert_int_eq(12287, sb.st_size);
    ck_assert_int_eq(1565283152, sb.st_ctime);
    ck_assert_int_eq(1565283167, sb.st_mtime);

    status = fs_ops.getattr("/file.8k+", &sb);
    ck_assert_int_eq(0, status);
    ck_assert_int_eq(500, sb.st_uid);
    ck_assert_int_eq(500, sb.st_gid);
    ck_assert_int_eq(0100666, sb.st_mode);
    ck_assert_int_eq(8195, sb.st_size);
    ck_assert_int_eq(1565283152, sb.st_ctime);
    ck_assert_int_eq(1565283167, sb.st_mtime);

    // Test error output
    int err;
    err = fs_ops.getattr("/not-a-file", &sb);
    ck_assert_int_eq(-ENOENT, err);

    err = fs_ops.getattr("/file.1k/file.0", &sb);
    ck_assert_int_eq(-ENOTDIR, err);

    err = fs_ops.getattr("/not-a-dir/file.0", &sb);
    ck_assert_int_eq(-ENOENT, err);

    err = fs_ops.getattr("/dir2/not-a-file", &sb);
    ck_assert_int_eq(-ENOENT, err);
}
END_TEST

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

void mark_seen(struct seen_file dir_table[], int count) {
    for (int i = 0; i < count; i++) {
        for (int j = 0; j < count; j++) {
            if (strcmp(entries[i], dir_table[j].name) == 0) {
                dir_table[j].seen = 1;
            }
        }
    }
}

void validate_seen(struct seen_file dir_table[], int count) {
    for (int i = 0; i < count; i++) {
        ck_assert_int_eq(1, dir_table[i].seen);
    }    
}

START_TEST(test_readdir)
{
    int count;

    init_entries();
    fs_ops.readdir("/",  NULL, empty_filler, 0, NULL);
    count = 6;
    ck_assert_int_eq(count, entryc);
    mark_seen(dir1_table, count);
    validate_seen(dir1_table, count);

    init_entries();
    fs_ops.readdir("/dir2",  NULL, empty_filler, 0, NULL);
    count = 2;
    ck_assert_int_eq(count, entryc);
    mark_seen(dir1_table, count);
    validate_seen(dir1_table, count);

    init_entries();
    fs_ops.readdir("/dir3",  NULL, empty_filler, 0, NULL);
    count = 2;
    ck_assert_int_eq(count, entryc);
    mark_seen(dir1_table, count);
    validate_seen(dir1_table, count);

    init_entries();
    fs_ops.readdir("/dir3/subdir",  NULL, empty_filler, 0, NULL);
    count = 3;
    ck_assert_int_eq(count, entryc);
    mark_seen(dir1_table, count);
    validate_seen(dir1_table, count);

    init_entries();
    fs_ops.readdir("dir-with-long-name",  NULL, empty_filler, 0, NULL);
    count = 1;
    ck_assert_int_eq(count, entryc);
    mark_seen(dir1_table, count);
    validate_seen(dir1_table, count);

    // test error
    int err;
    err = fs_ops.readdir("/not-a-dir",  NULL, empty_filler, 0, NULL);
    ck_assert_int_eq(err, -ENOENT);

    err = fs_ops.readdir("/file.1k",  NULL, empty_filler, 0, NULL);
    ck_assert_int_eq(err, -ENOTDIR);

    err = fs_ops.readdir("/not-a-dir/file.0",  NULL, empty_filler, 0, NULL);
    ck_assert_int_eq(err, -ENOENT);

    err = fs_ops.readdir("/dir2/not-a-file",  NULL, empty_filler, 0, NULL);
    ck_assert_int_eq(err, -ENOENT);
}
END_TEST

// Read file n bytes at a time.
// Return total bytes read for this file.
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

START_TEST(test_read)
{
    int status;

    // Test read error;
    status = fs_ops.read("/dir2", NULL, 4096, 0, NULL);
    ck_assert_int_eq(status, -EISDIR);

    status = fs_ops.read("/dir2/no-exist-file", NULL, 4096, 0, NULL);
    ck_assert_int_eq(status, -ENOENT);


    // Test read the file with different N value as stated in the Project description.
    int N[9] = {17, 100, 1000, 1024, 1970, 3000, 5000, 10000, 20000};
    for (int i = 0; i < 9; i++) {
            char buf[20480];
            int len;
            unsigned int cksum;

            len = read_file_by_n("/file.1k", buf, N[i]);
            cksum = crc32(0, (const unsigned char *) buf, len);
            ck_assert_int_eq(len, 1000);
            ck_assert_uint_eq(cksum, 1786485602);

            len = read_file_by_n("/file.10", buf, N[i]);
            cksum = crc32(0, (const unsigned char *) buf, len);
            ck_assert_int_eq(len, 10);
            ck_assert_uint_eq(cksum, 855202508);

            len = read_file_by_n("/dir-with-long-name/file.12k+", buf, N[i]);
            cksum = crc32(0, (const unsigned char *) buf, len);
            ck_assert_int_eq(len, 12289);
            ck_assert_uint_eq(cksum, 4101348955);

            len = read_file_by_n("/dir2/twenty-seven-byte-file-name", buf, N[i]);
            cksum = crc32(0, (const unsigned char *) buf, len);
            ck_assert_int_eq(len, 1000);
            ck_assert_uint_eq(cksum, 2575367502);

            len = read_file_by_n("/dir2/file.4k+", buf, N[i]);
            cksum = crc32(0, (const unsigned char *) buf, len);
            ck_assert_int_eq(len, 4098);
            ck_assert_uint_eq(cksum, 799580753);

            len = read_file_by_n("/dir3/subdir/file.4k-", buf, N[i]);
            cksum = crc32(0, (const unsigned char *) buf, len);
            ck_assert_int_eq(len, 4095);
            ck_assert_uint_eq(cksum, 4220582896);

            len = read_file_by_n("/dir3/subdir/file.8k-", buf, N[i]);
            cksum = crc32(0, (const unsigned char *) buf, len);
            ck_assert_int_eq(len, 8190);
            ck_assert_uint_eq(cksum, 4090922556);

            len = read_file_by_n("/dir3/subdir/file.12k", buf, N[i]);
            cksum = crc32(0, (const unsigned char *) buf, len);
            ck_assert_int_eq(len, 12288);
            ck_assert_uint_eq(cksum, 3243963207);

            len = read_file_by_n("/dir3/file.12k-", buf, N[i]);
            cksum = crc32(0, (const unsigned char *) buf, len);
            ck_assert_int_eq(len, 12287);
            ck_assert_uint_eq(cksum, 2954788945);

            len = read_file_by_n("/file.8k+", buf, N[i]);
            cksum = crc32(0, (const unsigned char *) buf, len);
            ck_assert_int_eq(len, 8195);
            ck_assert_uint_eq(cksum, 2112223143);
    }
}
END_TEST

START_TEST(test_statfs)
{
    struct statvfs st;
    fs_ops.statfs(NULL, &st);
    ck_assert_int_eq(4096, st.f_bsize);
    ck_assert_int_eq(400, st.f_blocks);
    ck_assert_int_eq(355, st.f_bfree);
    ck_assert_int_eq(355, st.f_bavail);
    ck_assert_int_eq(27, st.f_namemax);

}
END_TEST

START_TEST(test_rename)
{
    int status;
    struct stat sb;
    status = fs_ops.rename("/file.1k", "file.1k.renamed");
    ck_assert_int_eq(0, status);
    ck_assert_int_eq(-ENOENT, fs_ops.getattr("/file.1k", &sb));
    fs_ops.getattr("/file.1k.renamed", &sb);
    ck_assert_int_eq(500, sb.st_uid);
    ck_assert_int_eq(500, sb.st_gid);
    ck_assert_int_eq(0100666, sb.st_mode);
    ck_assert_int_eq(1000, sb.st_size);
    ck_assert_int_eq(1565283152, sb.st_ctime);
    ck_assert_int_eq(1565283152, sb.st_mtime);

    status = fs_ops.rename("/dir3/subdir", "/dir3/subdir_renamed");
    ck_assert_int_eq(0, status);
    ck_assert_int_eq(-ENOENT, fs_ops.getattr("/dir3/subdir", &sb));
    fs_ops.getattr("/dir3/subdir_renamed", &sb);
    ck_assert_int_eq(0, sb.st_uid);
    ck_assert_int_eq(500, sb.st_gid);
    ck_assert_int_eq(040777, sb.st_mode);
    ck_assert_int_eq(4096, sb.st_size);
    ck_assert_int_eq(1565283152, sb.st_ctime);
    ck_assert_int_eq(1565283167, sb.st_mtime);

    status = fs_ops.rename("/dir3/src_not_exist", "/dir3/something");
    ck_assert_int_eq(-ENOENT, status);
    fs_ops.getattr("dir3",  &sb);
    ck_assert_int_ne(1565283167, sb.st_mtime);

    status = fs_ops.rename("/dir3", "/dir3/something");
    ck_assert_int_eq(-EINVAL, status);

    status = fs_ops.rename("/file.10", "/file.8k+");
    ck_assert_int_eq(-EEXIST, status);

    status = fs_ops.rename("/dir3/subdir_renamed/file.4k-", "/dir3/subdir_renamed/file.8k-");
    ck_assert_int_eq(-EEXIST, status);

    status = fs_ops.rename("/dir-with-long-name/file.12k+", "/dir-with-long-name/file.12k+");
    ck_assert_int_eq(0, status);
}
END_TEST

START_TEST(test_chmod)
{
    int status;
    struct stat sb;
    status = fs_ops.chmod("/file.10", 0777);
    ck_assert_int_eq(0, status);
    fs_ops.getattr("/file.10", &sb);
    ck_assert_int_eq(500, sb.st_uid);
    ck_assert_int_eq(500, sb.st_gid);
    ck_assert_int_eq(0100777, sb.st_mode);
    ck_assert_int_eq(10, sb.st_size);
    ck_assert_int_eq(1565283152, sb.st_ctime);
    ck_assert_int_eq(1565283167, sb.st_mtime);

    status = fs_ops.chmod("/dir3", 0555);
    ck_assert_int_eq(0, status);
    fs_ops.getattr("/dir3", &sb);
    ck_assert_int_eq(0, sb.st_uid);
    ck_assert_int_eq(500, sb.st_gid);
    ck_assert_int_eq(040555, sb.st_mode);
    ck_assert_int_eq(4096, sb.st_size);
    ck_assert_int_eq(1565283152, sb.st_ctime);
    ck_assert_int_eq(1565283167, sb.st_mtime);

    // Test error output
    int err;
    err = fs_ops.chmod("/not-a-file", 0777);
    ck_assert_int_eq(-ENOENT, err);

    err = fs_ops.chmod("/file.8k+/file.0", 0777);
    ck_assert_int_eq(-ENOTDIR, err);

    err = fs_ops.chmod("/not-a-dir/file.0", 0777);
    ck_assert_int_eq(-ENOENT, err);

    err = fs_ops.chmod("/dir2/not-a-file", 0777);
    ck_assert_int_eq(-ENOENT, err);
}
END_TEST

int main(int argc, char **argv)
{
    block_init("test.img");
    fs_ops.init(NULL);
    
    Suite *s = suite_create("fs5600");
    TCase *tc = tcase_create("read_mostly");

    tcase_add_test(tc, test_getattr); 
    tcase_add_test(tc, test_readdir); 
    tcase_add_test(tc, test_read); 
    tcase_add_test(tc, test_statfs); 
    tcase_add_test(tc, test_chmod); 
    tcase_add_test(tc, test_rename); 
    
    suite_add_tcase(s, tc);
    SRunner *sr = srunner_create(s);
    srunner_set_fork_status(sr, CK_NOFORK);
    
    srunner_run_all(sr, CK_VERBOSE);
    int n_failed = srunner_ntests_failed(sr);
    printf("%d tests failed\n", n_failed);
    
    srunner_free(sr);
    return (n_failed == 0) ? EXIT_SUCCESS : EXIT_FAILURE;
    return 0;
}
