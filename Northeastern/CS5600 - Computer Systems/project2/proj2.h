/*
 * file:        proj2.h
 * description: request/response header for project 2
 */
#ifndef __PROJ2_H__
#define __PROJ2_H__
#define MAX_KEY_NUM 200
#define MAX_KEY_SIZE 30

struct request {
    char op_status;             /* R/W/D, K/X */
    char name[31];              /* null-padded, max strlen = 30 */
    char len[8];                /* text, decimal, null-padded */
};

#endif
