#include <iostream>
#include <cstdlib>

int main(int argc, char *argv[]) {

    int parameter = std::atoi(argv[1]);

    if (parameter == 0) {
        // Execute executable_A
        if (std::system("./subtask3") != 0) {
            std::cerr << "Error executing without streams" << std::endl;
            return 1;
        }
    } else {
        // Execute executable_B
        if (std::system("./streams") != 0) {
            std::cerr << "Error executing with streams" << std::endl;
            return 1;
        }
    }

    return 0;
}