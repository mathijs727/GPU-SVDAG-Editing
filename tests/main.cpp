#define CATCH_CONFIG_RUNNER
#include <catch2/catch_all.hpp>

int main(int argc, char* argv[])
{
    // your setup ...

    int result = Catch::Session().run(argc, argv);

    // your clean-up...

    return result;
}