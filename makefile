.PHONY: clean all subtask1 subtask2 subtask3 subtask4

all: subtask1 subtask2 run_python_script subtask3 subtask4

subtask1:
	g++ -std=c++17 -O3 src/assignment2_subtask1.cpp -o subtask1

subtask2: 
	nvcc -O3 src/assignment2_subtask2.cu -o subtask2

run_python_script:
	python3 preprocess.py

subtask3: 
	nvcc -O3 src/assignment2_subtask3.cu -o subtask3

subtask4:
	nvcc -O3 src/assignment2_subtask4.cu -o streams
	g++ -std=c++17 src/proxy.cpp -o subtask4

clean:
	rm subtask1 subtask2 subtask3 subtask4 streams
	rm output/*
	rm pre-proc-img/*