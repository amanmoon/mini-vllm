#!/usr/bin/env bash

set -e

BUILD_DIR="build"
MAIN_TARGET="transformer"

configure() {
    cmake -S . -B "$BUILD_DIR"
}

build_target() {
    local TARGET="$1"

    echo "Configuring project..."
    configure

    echo "Building $TARGET..."
    cmake --build "$BUILD_DIR" --target "$TARGET" -- -j"$(nproc)"
}

run_target() {
    local TARGET="$1"
    local EXECUTABLE="$BUILD_DIR/$TARGET"

    if [ ! -x "$EXECUTABLE" ]; then
        echo "Executable '$TARGET' not found."
        echo "Run './minivllm.sh build $TARGET' first."
        exit 1
    fi

    echo "========================================"
    echo "Running $TARGET"
    echo "========================================"

    "$EXECUTABLE"
}

run_all_tests() {
    FOUND=0

    for test in "$BUILD_DIR"/test_*; do
        if [ -x "$test" ]; then
            FOUND=1
            echo "========================================"
            echo "Running $(basename "$test")"
            echo "========================================"
            "$test"
            echo
        fi
    done

    if [ "$FOUND" -eq 0 ]; then
        echo "No test executables found."
    fi
}

case "$1" in

    build)

        # build
        if [ $# -eq 1 ]; then
            build_target "$MAIN_TARGET"

        # build test
        elif [ "$2" = "test" ]; then
            echo "Configuring project..."
            configure

            echo "Building all tests..."

            for target in $(find tests -type f \( -name "*.cpp" -o -name "*.cu" \) \
                | sed 's|tests/||' \
                | sed 's|\.[^.]*$||'); do

                cmake --build "$BUILD_DIR" --target "$target" -- -j"$(nproc)"
            done

            echo "✓ All tests built."

        # build run
        elif [ "$2" = "run" ]; then

            if [ $# -eq 2 ]; then
                build_target "$MAIN_TARGET"
                run_target "$MAIN_TARGET"
            else
                build_target "$3"
                run_target "$3"
            fi

        # build <target>
        else
            build_target "$2"
        fi
        ;;

    run)

        # run
        if [ $# -eq 1 ]; then
            run_target "$MAIN_TARGET"

        # run test
        elif [ "$2" = "test" ]; then
            run_all_tests

        # run <target>
        else
            run_target "$2"
        fi
        ;;

    clean)
        rm -rf "$BUILD_DIR"
        echo "✓ Build directory removed."
        ;;

    *)
        echo "Usage:"
        echo
        echo "  ./minivllm.sh build"
        echo "      Build the main executable."
        echo
        echo "  ./minivllm.sh build test"
        echo "      Build all test executables."
        echo
        echo "  ./minivllm.sh build <test_name>"
        echo "      Build one test executable."
        echo
        echo "  ./minivllm.sh build run"
        echo "      Build and run the main executable."
        echo
        echo "  ./minivllm.sh build run <test_name>"
        echo "      Build and run one test executable."
        echo
        echo "  ./minivllm.sh run"
        echo "      Run the main executable."
        echo
        echo "  ./minivllm.sh run test"
        echo "      Run all test executables."
        echo
        echo "  ./minivllm.sh run <test_name>"
        echo "      Run one test executable."
        echo
        echo "  ./minivllm.sh clean"
        exit 1
        ;;
esac