import os
import torch
import torchaudio

AUDIO_DIR = "audio"
RTTM_DIR = "output_rttm"

def parse_rttm_line(line):
	parts = line.strip().split()
	file_id = parts[1]
	start = float(parts[3])
	duration = float(parts[4]) #seg duration
	end = start + duration 
	speaker = parts[7]
	return file_id, start, end, speaker

# convert rttm to tensor
def rttm_to_tensors(rttm_file, audio_dir):
	tensors = []
	labels = []
	
	file_id = os.path.basename(rttm_file).replace(".rttm", "")
	audio_path = os.path.join(AUDIO_DIR, f"{file_id}.wav")
	
	if not os.path.exists(audio_path):
		print(f"missing audio file for {file_id}")
		return [], []

	waveform, sample_rate = torchaudio.load(audio_path)
	
	with open(rttm_file, "r") as f:
		for line in f:
			if line.startswith("SPEAKER"):
				_, start, end, speaker = parse_rttm_line(line)  #skip file id
				start_sample = int(start * sample_rate)
				end_sample = int(end * sample_rate)
				segment = waveform[:, start_sample:end_sample] #slice tensor
				tensors.append(segment)
				labels.append(speaker)
	return tensors, labels





#ex
all_tensors = []
all_labels = []

for fname in os.listdir(RTTM_DIR):
	if fname.endswith(".rttm"):
		rttm_path = os.path.join(RTTM_DIR, fname)
		tensors, labels = rttm_to_tensors(rttm_path, AUDIO_DIR)
		all_tensors.extend(tensors)
		all_labels.extend(labels)

print(f"total segments: {len(all_tensors)}")

# print(f"example tensor shape: {all_tensors[0].shape}")

# print ex tensor 
for i in range(10):
	print(f"segment {i}: shape={all_tensors[i].shape}, label={all_labels[i]}")

