/*******************************************************************************
* Copyright 2018 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <iostream>

using namespace std;

struct Scales {
    string name;
    float scale_out;
    struct Scales *next;
};

struct Scales* init_scale() {
    Scales *root = new Scales();
    root->next = NULL;
    root->scale_out = 0.0;

    return root;
}

struct Scales* push_scale(struct Scales *node, string name, float scale_out) {
    Scales *newNode = new Scales();
    newNode->next = NULL;
    newNode->name = name;
    newNode->scale_out = scale_out;
    node->next = newNode;

    return newNode;
}

void PrintAll(struct Scales *p) {
    while( p->next ) {
        p = p->next;  
        printf( "%s: %f\n", p->name.c_str(), p->scale_out );
    }
}

float get_scales(struct Scales *root, string name) {
    struct Scales *node = root;
    while( node->next ) {
        node = node->next;  
        if ( node->name.c_str() == name ) return node->scale_out;
    }

    return -1.0;
}
