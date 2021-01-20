
#include <gtest/gtest.h>

#include <cstdlib>
#include <cuda_runtime.h>

// Global variables to store the command line parameters:
int cBDAGTestArgC;
char** cBDAGTestArgV;

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);


    // store the argc and argv command line parameters to global variables
    // so they can be retrieved in other code files (inside the tests):
    cBDAGTestArgC = argc;
    cBDAGTestArgV = argv;

    int vResult = RUN_ALL_TESTS();

    return vResult;
}
