import sys

with open(sys.argv[1], "r") as pred:
	pred_lines = list(pred.readlines())

with open("output/gold_file.txt", "r") as filenames:
	idx = 0
	for fname in filenames:
		cur_name = fname.rstrip()
		with open("output/" + cur_name[:11] + ".txt", "w") as new_file:
			for L in pred_lines[idx:idx+7]:
				new_file.write(L)
		idx += 7