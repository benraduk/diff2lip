import subprocess
import sys
import os

# Set environment variables for single-process execution
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["MASTER_ADDR"] = "localhost"  
os.environ["MASTER_PORT"] = "12355"
os.environ["RANK"] = "0"
os.environ["WORLD_SIZE"] = "1"

# Command arguments
cmd = [
    "python", "generate.py",
    "--attention_resolutions", "32,16,8",
    "--class_cond", "False",
    "--learn_sigma", "True", 
    "--num_channels", "128",
    "--num_head_channels", "64",
    "--num_res_blocks", "2",
    "--resblock_updown", "True",
    "--use_fp16", "True",
    "--use_scale_shift_norm", "False",
    "--predict_xstart", "False",
    "--diffusion_steps", "1000",
    "--noise_schedule", "linear",
    "--rescale_timesteps", "False",
    "--sampling_seed", "7",
    "--sampling_input_type", "gt",
    "--sampling_ref_type", "gt", 
    "--timestep_respacing", "ddim25",
    "--use_ddim", "True",
    "--model_path", "checkpoints/checkpoint.pt",
    "--nframes", "5",
    "--nrefer", "1",
    "--image_size", "128",
    "--sampling_batch_size", "8",
    "--face_hide_percentage", "0.5",
    "--use_ref", "True",
    "--use_audio", "True",
    "--audio_as_style", "True",
    "--generate_from_filelist", "0",
    "--video_path", "test_media/person.mp4",
    "--audio_path", "test_media/speech.m4a",
    "--out_path", "output_dir/result.mp4",
    "--save_orig", "False",
    "--face_det_batch_size", "16",
    "--pads", "0,0,0,0",
    "--is_voxceleb2", "False"
]

# Run the command
try:
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    print("Success!")
    print("STDOUT:")
    print(result.stdout)
    print("STDERR:")
    print(result.stderr)
except subprocess.CalledProcessError as e:
    print("Error occurred:")
    print("STDOUT:")
    print(e.stdout)
    print("STDERR:")
    print(e.stderr)