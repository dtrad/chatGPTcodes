// create a hello world c program
// compile it with gcc
// convert to assembly
// gcc -S testassembly.c
// open testassembly.s
// compile assembly to object file
// gcc -c testassembly.s
// link object file to create executable

// create a function to sum two numbers
#include <stdio.h>
int sum(int a, int b) {
    return a + b;
}
void main() {
    int a = 5;
    int b = 10;
    int result = sum(a, b);
    printf("The sum of %d and %d is %d", a, b, result);
}


