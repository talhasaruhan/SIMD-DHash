# SIMD-DHash
SIMD DHash implementation in C/C++


Header only, include DHash.h and you're good to go!

There are 4 different implementations for Resize function that take an image and resizes it down to 8x8.
Then the DHash function copies the first row to the 9th row and perform a row comparison and compresses the data down to an uint64_t.

test.cpp does include the tests for all the functions along with the naive, easy to follow implementations.
