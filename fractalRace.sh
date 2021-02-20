echo "Running CPU fractal maker:\n"
time ./bin/serialFractal
echo "\nRunning GPU fractal maker:\n"
time ./bin/cudaFractal 1 1 -1.5 -.5 800 800 40
# args added to CUDA fractal to make params the same
