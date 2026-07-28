#!/usr/bin/env bash

set -e

BUILD_DIR="build"
EXECUTABLE="$BUILD_DIR/transformer"

case "$1" in
    build)
        echo "Cleaning previous build..."
        rm -rf "$BUILD_DIR"

        echo "Configuring project..."
        cmake -S . -B "$BUILD_DIR"

        echo "Building..."
        cmake --build "$BUILD_DIR" -- -j"$(nproc)"

        echo "✓ Build completed successfully."
        ;;

    run)
        if [ ! -f "$EXECUTABLE" ]; then
            echo "Executable not found."
            echo "Run './minivllm.sh build' first."
            exit 1
        fi

        echo "Running..."
        "$EXECUTABLE"
        ;;

    clean)
        rm -rf "$BUILD_DIR"
        echo "✓ Build directory removed."
        ;;

    *)
        echo "Usage: $0 {build|run|clean}"
        exit 1
        ;;
esac