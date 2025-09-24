import os


def saveParams(prefix, output, need_scaling_flag, need_preprocessing_flag, pixel_size_raw,density_thresh,clean_background_flag,min_size,patch_size,pixel_size):

	# Output file path
	output_dir = f"{prefix}{output}"
	os.makedirs(output_dir, exist_ok=True)
	param_file = os.path.join(output_dir, "HistoSweep_parameters.txt")

	# Write parameters to file
	with open(param_file, "w") as f:
	    f.write("===== USER-DEFINED INPUT PARAMETERS =====\n")
	    f.write(f"need_scaling_flag: {need_scaling_flag}\n")
	    f.write(f"need_preprocessing_flag: {need_preprocessing_flag}\n")
	    f.write(f"pixel_size_raw: {pixel_size_raw}\n")
	    f.write(f"density_thresh: {density_thresh}\n")
	    f.write(f"clean_background_flag: {clean_background_flag}\n")
	    f.write(f"min_size: {min_size}\n\n")
	    f.write("===== ADDITIONAL PARAMETERS =====\n")
	    f.write(f"patch_size: {patch_size}\n")
	    f.write(f"pixel_size: {pixel_size}\n")

	print(f"✅ Parameters saved to: {param_file}")
