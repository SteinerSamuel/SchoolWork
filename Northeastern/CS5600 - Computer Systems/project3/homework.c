/*
 * file: homework.c
 * description: skeleton file for CS 5600 system
 *
 * CS 5600, Computer Systems, Northeastern
 * Created by: Peter Desnoyers, November 2019
 */

#define FUSE_USE_VERSION 27
#define _FILE_OFFSET_BITS 64

#include <stdlib.h>
#include <stddef.h>
#include <unistd.h>
#include <fuse.h>
#include <fcntl.h>
#include <string.h>
#include <stdio.h>
#include <errno.h>

#include "fs5600.h"

#define MAX_PATH_LEN 10
#define MAX_NAME_LEN 27
#define MAX_ENTRY_NUM 128

/* if you don't understand why you can't use these system calls here, 
 * you need to read the assignment description another time
 */
#define stat(a, b) error do not use stat()
#define open(a, b) error do not use open()
#define read(a, b, c) error do not use read()
#define write(a, b, c) error do not use write()

/* disk access. All access is in terms of 4KB blocks; read and
 * write functions return 0 (success) or -EIO.
 */
extern int block_read(void *buf, int lba, int nblks);
extern int block_write(void *buf, int lba, int nblks);

// Helper functions that are factored out.
void inode_to_stat(struct fs_inode *inode, struct stat *sb);
int parse(const char *path, char **argv);
int translate(int pathc, char **pathv);

/* bitmap functions
 */
void bit_set(unsigned char *map, int i)
{
    map[i / 8] |= (1 << (i % 8));
    block_write((void*)map, 1, 1);
}
void bit_clear(unsigned char *map, int i)
{
    map[i / 8] &= ~(1 << (i % 8));
    block_write((void*)map, 1, 1);

}
int bit_test(unsigned char *map, int i)
{
    return map[i / 8] & (1 << (i % 8));
}

struct fs_super super;
unsigned char bitmap[FS_BLOCK_SIZE];
struct fs_inode root;

void inode_to_stat(struct fs_inode *inode, struct stat *sb) {
    sb->st_uid = inode->uid;
    sb->st_gid = inode->gid;
    sb->st_mode = inode->mode;
    sb->st_size = inode->size;
    sb->st_ctime = inode->ctime;
    sb->st_mtime = inode->mtime;
    sb->st_atime = inode->mtime;
    sb->st_nlink = 1;
}

int parse(const char *c_path, char **argv) {
    int i;
    char *path = strdup(c_path);
    for (i = 0; i < MAX_PATH_LEN; i++) {
        if ((argv[i] = strtok(path, "/")) == NULL) {
            break;
        }
        if (strlen(argv[i]) > MAX_NAME_LEN) {
            argv[i][MAX_NAME_LEN] = 0; // truncate to 27 characters
        }
        path = NULL;
    }
    free(path);
    return i;
}

/* convert path to inode number.
 */
int translate(int pathc, char **pathv) {
    int inum = 2;
    struct fs_inode inode;
    for (int i = 0; i < pathc; i++) {
        int found = 0;
        block_read((void*)&inode, inum, 1);

        if (!S_ISDIR(inode.mode)) {
            return -ENOTDIR;
        }

        for (int j = 0; j < inode.size / FS_BLOCK_SIZE + 1; j++) {
            if (inode.ptrs[j] > 0) {
                struct fs_dirent entries[MAX_ENTRY_NUM] = {{0}};
                block_read((void*)&entries, inode.ptrs[j], 1);
                for (int k = 0; k < MAX_ENTRY_NUM; k++) {
                    if (strcmp(entries[k].name, pathv[i]) == 0 && entries[k].valid != 0) {
                        inum = entries[k].inode;
                        found = 1;
                        break;
                    }
                }
            }
            if (found == 1) {
                break;
            }
        }

        if (found == 0) {
            return -ENOENT;
        }
    }

    return inum;
}

int count_free_blocks() {
    int count = 0;
    for (int i = 0; i < super.disk_size; i++) {
        if (bit_test(bitmap, i) == 0) {
            count +=1;
        }
    }
    return count;
}


int find_first_free_block() {
    for (int i = 0; i < super.disk_size; i++) {
        if (bit_test(bitmap, i) == 0) {
            return i;
        }
    }
    return -1;
}

/* init - this is called once by the FUSE framework at startup. Ignore
 * the 'conn' argument.
 * recommended actions:
 *   - read superblock
 *   - allocate memory, read bitmaps and inodes
 */
void *fs_init(struct fuse_conn_info *conn)
{
    /* your code here */
    block_read((void*)&super, 0, 1);
    block_read((void*)bitmap, 1, 1);
    block_read((void*)&root, 2, 1);

    return NULL;
}

/* Note on path translation errors:
 * In addition to the method-specific errors listed below, almost
 * every method can return one of the following errors if it fails to
 * locate a file or directory corresponding to a specified path.
 *
 * ENOENT - a component of the path doesn't exist.
 * ENOTDIR - an intermediate component of the path (e.g. 'b' in
 *           /a/b/c) is not a directory
 */

/* note on splitting the 'path' variable:
 * the value passed in by the FUSE framework is declared as 'const',
 * which means you can't modify it. The standard mechanisms for
 * splitting strings in C (strtok, strsep) modify the string in place,
 * so you have to copy the string and then free the copy when you're
 * done. One way of doing this:
 *
 *    char *_path = strdup(path);
 *    int inum = translate(_path);
 *    free(_path);
 */

/* getattr - get file or directory attributes. For a description of
 *  the fields in 'struct stat', see 'man lstat'.
 *
 * Note - for several fields in 'struct stat' there is no corresponding
 *  information in our file system:
 *    st_nlink - always set it to 1
 *    st_atime, st_ctime - set to same value as st_mtime
 *
 * success - return 0
 * errors - path translation, ENOENT
 * hint - factor out inode-to-struct stat conversion - you'll use it
 *        again in readdir
 */
int fs_getattr(const char *path, struct stat *sb)
{
    int inum;
    struct fs_inode inode;
    char *pathv[MAX_PATH_LEN];
    int  pathc = parse(path, pathv);
    // As stated in the project description:
    // with the standard convention that a negative number is an error
    inum = translate(pathc, pathv);
    if (inum < 0) {
        return inum;
    }
    
    block_read((void*)&inode, inum, 1);
    inode_to_stat(&inode, sb);

    return 0;
}

/* readdir - get directory contents.
 *
 * call the 'filler' function once for each valid entry in the 
 * directory, as follows:
 *     filler(buf, <name>, <statbuf>, 0)
 * where <statbuf> is a pointer to a struct stat
 * success - return 0
 * errors - path resolution, ENOTDIR, ENOENT
 * 
 * hint - check the testing instructions if you don't understand how
 *        to call the filler function
 */
int fs_readdir(const char *path, void *ptr, fuse_fill_dir_t filler,
               off_t offset, struct fuse_file_info *fi)
{
    int inum;
    struct fs_inode inode;
    char *pathv[MAX_PATH_LEN];
    int  pathc = parse(path, pathv);
    inum = translate(pathc, pathv);
    if (inum < 0) {
        return inum;
    }
    
    block_read((void*)&inode, inum, 1);
    if (!S_ISDIR(inode.mode)) {
        return -ENOTDIR;
    }

    for (int i = 0; i < inode.size / FS_BLOCK_SIZE + 1; i++) {
        if (inode.ptrs[i] > 0) {
            struct fs_dirent entries[MAX_ENTRY_NUM] = {{0}};
            block_read((void*)&entries, inode.ptrs[i], 1);
            for (int j = 0; j < MAX_ENTRY_NUM; j++) {
                if (entries[j].valid != 0) {
                    struct stat sb;
                    char fullpath[MAX_PATH_LEN * MAX_PATH_LEN];
                    sprintf(fullpath, "%s%s", path, entries[j].name);
                    fs_getattr(fullpath, &sb);
                    filler(NULL, entries[j].name, &sb, 0);
                }
            }
        }
    }

    return 0;
}

/* create - create a new file with specified permissions
 *
 * success - return 0
 * errors - path resolution, EEXIST
 *          in particular, for create("/a/b/c") to succeed,
 *          "/a/b" must exist, and "/a/b/c" must not.
 *
 * Note that 'mode' will already have the S_IFREG bit set, so you can
 * just use it directly. Ignore the third parameter.
 *
 * If a file or directory of this name already exists, return -EEXIST.
 * If there are already 128 entries in the directory (i.e. it's filled an
 * entire block), you are free to return -ENOSPC instead of expanding it.
 */
int fs_create(const char *path, mode_t mode, struct fuse_file_info *fi)
{
    int inum;
    int pnum;
    int newnum;
    struct fs_inode inode;
    struct fs_inode nnode;
    struct fuse_context *ctx = fuse_get_context();

    char *pathv[MAX_PATH_LEN];
    int  pathc = parse(path, pathv);
    char *file_name = pathv[pathc-1];
    inum = translate(pathc, pathv);
    if (inum >= 0) {
        return -EEXIST;
    }

    pnum = translate(pathc-1, pathv);
    if (pnum < 0) {
        return pnum;
    }

    block_read((void*)&inode, pnum, 1);

    if (!S_ISDIR(inode.mode)) {
        return -ENOTDIR;
    }

    //make the new node
    uint16_t uid = ctx->uid;
    uint16_t gid = ctx->gid; 
    nnode.uid = uid;
    nnode.gid = gid;
    nnode.mode = mode;
    nnode.ctime = time(NULL);
    nnode.mtime = time(NULL);
    nnode.size = 0;


    for (int i = 0; i < inode.size / FS_BLOCK_SIZE + 1; i++) {
        if (inode.ptrs[i] > 0) {
            struct fs_dirent entries[MAX_ENTRY_NUM] = {{0}};
            block_read((void*)&entries, inode.ptrs[i], 1);
            for (int j = 0; j < MAX_ENTRY_NUM; j++) {
                if (entries[j].valid == 0) {
                    newnum = find_first_free_block();
                    if (newnum < 0) {
                        return -ENOSPC;
                    }
                    block_write((void*)&nnode, newnum, 1);
                    bit_set(bitmap, newnum);

                    entries[j].valid = 1;
                    strcpy(entries[j].name, file_name);
                    entries[j].inode = newnum;
                    block_write((void*)&entries, inode.ptrs[i], 1);
                    return 0;
                }
            }
        }
    }

    return -ENOSPC;
}

/* mkdir - create a directory with the given mode.
 *
 * WARNING: unlike fs_create, @mode only has the permission bits. You
 * have to OR it with S_IFDIR before setting the inode 'mode' field.
 *
 * success - return 0
 * Errors - path resolution, EEXIST
 * Conditions for EEXIST are the same as for create. 
 */
int fs_mkdir(const char *path, mode_t mode)
{
    int inum;
    int pnum;
    int newnum;
    struct fs_inode inode;
    struct fs_inode nnode;
    struct fuse_context *ctx = fuse_get_context();

    char *pathv[MAX_PATH_LEN];
    int  pathc = parse(path, pathv);
    char *file_name = pathv[pathc-1];
    inum = translate(pathc, pathv);
    if (inum >= 0) {
        return -EEXIST;
    }

    pnum = translate(pathc-1, pathv);
    if (pnum < 0) {
        return pnum;
    }

    block_read((void*)&inode, pnum, 1);

    if (!S_ISDIR(inode.mode)) {
        return -ENOTDIR;
    }
    
    int bnum = find_first_free_block();

    if (bnum < 0) {
        return -ENOSPC;
    }
    bit_set(bitmap, bnum);
    //make the new node
    uint16_t uid = ctx->uid;
    uint16_t gid = ctx->gid; 
    nnode.uid = uid;
    nnode.gid = gid;
    nnode.mode = (S_IFDIR | mode);
    nnode.ctime = time(NULL);
    nnode.mtime = time(NULL);
    nnode.size = FS_BLOCK_SIZE;
    nnode.ptrs[0] =  bnum;

    for (int i = 0; i < inode.size / FS_BLOCK_SIZE + 1; i++) {
        if (inode.ptrs[i] > 0) {
            struct fs_dirent entries[MAX_ENTRY_NUM] = {{0}};
            struct fs_dirent newentries[MAX_ENTRY_NUM] = {{0}};
            block_write((void*)&newentries, bnum, 1);
            block_read((void*)&entries, inode.ptrs[i], 1);
            for (int j = 0; j < MAX_ENTRY_NUM; j++) {
                if (entries[j].valid == 0) {
                    newnum = find_first_free_block();
                    if (newnum < 0) {
                        //free the block that the directory is because we cant give it a block for the inum
                        bit_clear(bitmap, bnum); 
                        return -ENOSPC;
                    }
                    block_write((void*)&nnode, newnum, 1);
                    bit_set(bitmap, newnum);

                    entries[j].valid = 1;
                    strcpy(entries[j].name, file_name);
                    entries[j].inode = newnum;
                    block_write((void*)&entries, inode.ptrs[i], 1);
                    return 0;
                }
            }
        }
    }
    // if we get here that means the parent folder has reached max entries and therefor we need to release the bit
    bit_clear(bitmap, bnum);
    return -ENOSPC;
}

/* unlink - delete a file
 *  success - return 0
 *  errors - path resolution, ENOENT, EISDIR
 */
int fs_unlink(const char *path)
{
    int inum;
    int pnum;
    struct fs_inode inode;
    struct fs_inode pnode;
    char *pathv[MAX_PATH_LEN];
    int  pathc = parse(path, pathv);
    char *dir_name = pathv[pathc-1];
    inum = translate(pathc, pathv);
    if (inum < 0) {
        return inum;
    }
    
    block_read((void*)&inode, inum, 1);
    if (S_ISDIR(inode.mode)) {
        return -EISDIR;
    }

    pnum = translate(pathc-1, pathv);
    block_read((void*)&pnode, pnum, 1);

    truncate(path, 0); // mark all blocks being used by the file as clear this is the same as truncate which is why its used

    for (int i = 0; i < inode.size / FS_BLOCK_SIZE + 1; i++) {
        if (pnode.ptrs[i] > 0) {
            struct fs_dirent entries[MAX_ENTRY_NUM] = {{0}};
            block_read((void*)&entries, pnode.ptrs[i], 1);
            for (int j = 0; j < MAX_ENTRY_NUM; j++) {
                if (entries[j].valid != 0 && strcmp(entries[j].name, dir_name) == 0) {
                    entries[j].valid = 0;
                    block_write((void*)&entries, pnode.ptrs[i], 1);
                    bit_clear(bitmap, inum);
                    return 0;
                }
            }
        }
    }


    return -ENOENT;
}

/* rmdir - remove a directory
 *  success - return 0
 *  Errors - path resolution, ENOENT, ENOTDIR, ENOTEMPTY
 */
int fs_rmdir(const char *path)
{   
    //initialization
    int inum;
    int pnum;
    struct fs_inode inode;
    struct fs_inode pnode;
    char *pathv[MAX_PATH_LEN];
    int  pathc = parse(path, pathv);
    char *dir_name = pathv[pathc-1];
    // make sure the path is real and resolves to a directory
    inum = translate(pathc, pathv);
    if (inum < 0) {
        return inum;
    }
    
    block_read((void*)&inode, inum, 1);
    if (!S_ISDIR(inode.mode)) {
        return -ENOTDIR;
    }

    // once we confirm grab the parent
    pnum = translate(pathc-1, pathv);
    block_read((void*)&pnode, pnum, 1);
    
    // make sure the folder is empty and then clear the directory block
    for (int i = 0; i < inode.size / FS_BLOCK_SIZE + 1; i++) {
        if (inode.ptrs[i] > 0) {
            struct fs_dirent entries[MAX_ENTRY_NUM] = {{0}};
            block_read((void*)&entries, inode.ptrs[i], 1);
            for (int j = 0; j < MAX_ENTRY_NUM; j++) {
                if (entries[j].valid != 0) {
                    return -ENOTEMPTY;
                }
            }
            bit_clear(bitmap, inode.ptrs[i]);
        }
    }

    // remove the ptr from the parent and mark the bit as clear
    for (int i = 0; i < inode.size / FS_BLOCK_SIZE + 1; i++) {
        if (pnode.ptrs[i] > 0) {
            struct fs_dirent entries[MAX_ENTRY_NUM] = {{0}};
            block_read((void*)&entries, pnode.ptrs[i], 1);
            for (int j = 0; j < MAX_ENTRY_NUM; j++) {
                if (entries[j].valid != 0 && strcmp(entries[j].name, dir_name) == 0) {
                    entries[j].valid = 0;
                    block_write((void*)&entries, pnode.ptrs[i], 1);
                    bit_clear(bitmap, inum);
                    return 0;
                }
            }
        }
    }

    return -ENOENT;
}

/* rename - rename a file or directory
 * success - return 0
 * Errors - path resolution, ENOENT, EINVAL, EEXIST
 *
 * ENOENT - source does not exist
 * EEXIST - destination already exists
 * EINVAL - source and destination are not in the same directory
 *
 * Note that this is a simplified version of the UNIX rename
 * functionality - see 'man 2 rename' for full semantics. In
 * particular, the full version can move across directories, replace a
 * destination file, and replace an empty directory with a full one.
 */
int fs_rename(const char *src_path, const char *dst_path)
{
    if (strcmp(src_path, dst_path) == 0) {
        return 0;
    }

    int inum;
    struct fs_inode inode;
    char *src_pathv[MAX_PATH_LEN];
    int  src_pathc = parse(src_path, src_pathv);
    char *dst_pathv[MAX_PATH_LEN];
    int  dst_pathc = parse(dst_path, dst_pathv);
    char *src_name = src_pathv[src_pathc-1];
    char *dst_name = dst_pathv[dst_pathc-1];

    if (src_pathc != dst_pathc) {
        return -EINVAL;
    }

    inum = translate(src_pathc - 1, src_pathv);
    if (inum < 0) {
        return -ENOENT;
    }

    block_read(&inode, inum, 1);
    if (!S_ISDIR(inode.mode)) {
        return -ENOENT;
    }

    // check if dst_name exits.
    for (int i = 0; i < inode.size / FS_BLOCK_SIZE + 1; i++) {
        if (inode.ptrs[i] > 0) {
            struct fs_dirent entries[MAX_ENTRY_NUM] = {{0}};
            block_read((void*)&entries, inode.ptrs[i], 1);
            for (int j = 0; j < MAX_ENTRY_NUM; j++) {
                if (entries[j].valid != 0 && strcmp(entries[j].name, dst_name) == 0) {
                    return -EEXIST;
                }
            }
        }
    }

    for (int i = 0; i < inode.size / FS_BLOCK_SIZE + 1; i++) {
        if (inode.ptrs[i] > 0) {
            struct fs_dirent entries[MAX_ENTRY_NUM] = {{0}};
            block_read((void*)&entries, inode.ptrs[i], 1);
            for (int j = 0; j < MAX_ENTRY_NUM; j++) {
                if (entries[j].valid != 0 && strcmp(entries[j].name, src_name) == 0) {
                    strcpy(entries[j].name, dst_name);
                    block_write((void*)&entries, inode.ptrs[i], 1);
                    inode.mtime = time(NULL);
                    block_write((void*)&inode, inum, 1);
                    return 0;
                }
            }
        }
    }

    // src is not found if code reaches here.
    return -ENOENT;
}

/* chmod - change file permissions
 * utime - change access and modification times
 *         (for definition of 'struct utimebuf', see 'man utime')
 *
 * success - return 0
 * Errors - path resolution, ENOENT.
 */
int fs_chmod(const char *path, mode_t mode)
{
    int inum;
    struct fs_inode inode;
    char *pathv[MAX_PATH_LEN];
    int  pathc = parse(path, pathv);
    // As stated in the project description:
    // with the standard convention that a negative number is an error
    inum = translate(pathc, pathv);
    if (inum < 0) {
        return inum;
    }
    
    block_read((void*)&inode, inum, 1);
    inode.mode = (inode.mode & S_IFMT) | mode;
    block_write((void*)&inode, inum, 1);
    return 0;
}

int fs_utime(const char *path, struct utimbuf *ut)
{
    int inum;
    struct fs_inode inode;
    char *pathv[MAX_PATH_LEN];
    int  pathc = parse(path, pathv);
    // As stated in the project description:
    // with the standard convention that a negative number is an error
    inum = translate(pathc, pathv);
    if (inum < 0) {
        return inum;
    }
    // we only care about mod time in the utimbuf as our fs only keeps trac of this
    block_read((void*)&inode, inum, 1);
    inode.mtime = ut->modtime;
    block_write((void*)&inode, inum, 1);

    return 0;
}

/* truncate - truncate file to exactly 'len' bytes
 * success - return 0
 * Errors - path resolution, ENOENT, EISDIR, EINVAL
 *    return EINVAL if len > 0.
 */
int fs_truncate(const char *path, off_t len)
{
    /* you can cheat by only implementing this for the case of len==0,
     * and an error otherwise.
     */
    if (len != 0)
        return -EINVAL; /* invalid argument */

    /* your code here */
    int inum;
    struct fs_inode inode;
    char *pathv[MAX_PATH_LEN];
    int  pathc = parse(path, pathv);
    // As stated in the project description:
    // with the standard convention that a negative number is an error
    inum = translate(pathc, pathv);
    if (inum < 0) {
        return inum;
    }

    
    // read the block for the inodes
    block_read((void*)&inode, inum, 1);
        if (S_ISDIR(inode.mode)) {
        return -EISDIR;
    }
    for (int i = 0; i < ( inode.size - 1)/ FS_BLOCK_SIZE + 1; i++) {
        // clear the bit on the bitmap
        bit_clear(bitmap, inode.ptrs[i]);
        // reset the pointer
        inode.ptrs[i] = 0;
    }
    // set the size to 0 and the modification time
    inode.size = 0;
    inode.mtime = time(NULL);
    // write to the block
    block_write((void*)&inode, inum, 1);

    return 0;
}

/* read - read data from an open file.
 * success: should return exactly the number of bytes requested, except:
 *   - if offset >= file len, return 0
 *   - if offset+len > file len, return #bytes from offset to end
 *   - on error, return <0
 * Errors - path resolution, ENOENT, EISDIR
 */
int fs_read(const char *path, char *buf, size_t len, off_t offset,
            struct fuse_file_info *fi)
{
    int inum;
    struct fs_inode inode;
    char *pathv[MAX_PATH_LEN];
    int  pathc = parse(path, pathv);
    // As stated in the project description:
    // with the standard convention that a negative number is an error
    inum = translate(pathc, pathv);
    if (inum < 0) {
        return inum;
    }
    
    block_read((void*)&inode, inum, 1);
    if (S_ISDIR(inode.mode)) {
        return -EISDIR;
    }

    if (offset >= inode.size) {
        return 0;
    }

    if (offset + len > inode.size) {
        return inode.size - offset;
    }

    // bytes we've read
    int read_len = 0;
    char block_buf[FS_BLOCK_SIZE];

    // read the first block, which might not be an entire block.
    int block_offset = offset / FS_BLOCK_SIZE;
    // length in bytes that we should read in the first block based on the offset.
    int len_first_block = (block_offset + 1) * FS_BLOCK_SIZE - offset;
    if (len_first_block > len) {
        len_first_block = len;
    }
    block_read((void*)&block_buf, inode.ptrs[block_offset], 1);
    strncpy(buf + read_len, block_buf + (offset % FS_BLOCK_SIZE), len_first_block);
    read_len += len_first_block;
    block_offset += 1;

    // read the middle blocks, which are  complete blocks.
    while (read_len < (int)len - FS_BLOCK_SIZE) {
        block_read((void*)&block_buf, inode.ptrs[block_offset], 1);
        strncpy(buf + read_len, block_buf, FS_BLOCK_SIZE);
        read_len += FS_BLOCK_SIZE;
        block_offset += 1;
    }


    // read the last block, which might not be an entire block.
    // length in bytes that we should read in the last block.
    int len_last_block = len - read_len;
    if (len_last_block > 0) {
        block_read((void*)&block_buf, inode.ptrs[block_offset], 1);
        strncpy(buf + read_len, block_buf, len_last_block);
        read_len += len_last_block;
    }

    return read_len;
}

/* write - write data to a file
 * success - return number of bytes written. (this will be the same as
 *           the number requested, or else it's an error)
 * Errors - path resolution, ENOENT, EISDIR
 *  return EINVAL if 'offset' is greater than current file length.
 *  (POSIX semantics support the creation of files with "holes" in them, 
 *   but we don't)
 */
int fs_write(const char *path, const char *buf, size_t len,
             off_t offset, struct fuse_file_info *fi)
{
    /* your code here */
    int off_f = 0;
    int written = 0;
    int inum;
    struct fs_inode inode;
    char *pathv[MAX_PATH_LEN];
    int  pathc = parse(path, pathv);
    inum = translate(pathc, pathv);
    if (inum < 0) {
        return inum;
    }

    char temp[FS_BLOCK_SIZE];
    
    block_read((void*)&inode, inum, 1);
    if (S_ISDIR(inode.mode)) {
        return -EISDIR;
    }

    if (inode.size < offset) {
        return -EINVAL;
    }
    // itterate through each block needed to write from where the offset starts to the full len
    for (int i = offset/FS_BLOCK_SIZE;  i < (offset  + len-1)/FS_BLOCK_SIZE +1; i++) {   
        // check if the block is set
        if ((inode.size/FS_BLOCK_SIZE < i)| (inode.size == 0)) {
            // find the next block
            int ptr = find_first_free_block();
            if (ptr < 0) {
                return -ENOSPC;
            }
            // set the block 
            inode.ptrs[i] = ptr;
            bit_set(bitmap, ptr);
        }
        // read the block 
        block_read((void*)&temp, inode.ptrs[i], 1);
        // loop through each bit and set it
        for (int j = 0; j < FS_BLOCK_SIZE + 1; j++) {
            // if the offset hasnt been used skip to the offset
            if (off_f == 0){
                off_f ++;
                j = offset % FS_BLOCK_SIZE;
            }
            // copy into the temp buffsd
            temp[j] = (char)buf[written];
            written ++;
            // once we finished the buffer break
            if (written == len-1) {
                break;
            }
        }
        // write to the block
        block_write((void*)&temp, inode.ptrs[i], 1);
    }

    inode.size = offset+len;
    inode.mtime = time(NULL);
    block_write((void*)&inode, inum, 1);
    return len;
}

/* statfs - get file system statistics
 * see 'man 2 statfs' for description of 'struct statvfs'.
 * Errors - none. Needs to work.
 */
int fs_statfs(const char *path, struct statvfs *st)
{
    /* needs to return the following fields (set others to zero):
     *   f_bsize = BLOCK_SIZE
     *   f_blocks = total image - (superblock + block map)
     *   f_bfree = f_blocks - blocks used
     *   f_bavail = f_bfree
     *   f_namemax = <whatever your max namelength is>
     *
     * it's OK to calculate this dynamically on the rare occasions
     * when this function is called.
     */
    /* your code here */
    st->f_bsize = FS_BLOCK_SIZE;
    st->f_blocks = super.disk_size;
    st->f_bfree = count_free_blocks();
    st->f_bavail = st->f_bfree;
    st->f_namemax = MAX_NAME_LEN;

    st->f_files = 0;
    st->f_ffree = 0;
    st->f_fsid = 0;
    st->f_frsize = 0;
    st->f_flag = 0;
    st->f_favail = 0;

    return 0;
}

/* operations vector. Please don't rename it, or else you'll break things
 */
struct fuse_operations fs_ops = {
    .init = fs_init, /* read-mostly operations */
    .getattr = fs_getattr,
    .readdir = fs_readdir,
    .rename = fs_rename,
    .chmod = fs_chmod,
    .read = fs_read,
    .statfs = fs_statfs,

    .create = fs_create, /* write operations */
    .mkdir = fs_mkdir,
    .unlink = fs_unlink,
    .rmdir = fs_rmdir,
    .utime = fs_utime,
    .truncate = fs_truncate,
    .write = fs_write,
};
