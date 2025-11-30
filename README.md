# SpeechLab
the container files/the container are located in [`speechlab_diarization`](/speechlab_diarization)
... still working on testing it

to use you need to agree to these huggingface models:
- [segmentation 3.0](https://huggingface.co/pyannote/segmentation-3.0)
- [speaker diarization community 1](https://huggingface.co/pyannote/speaker-diarization-community-1)

note: vtc2.0 is [currently broken](https://github.com/LAAC-LSCP/VTC/issues/4) because the model weights are in a github lfs and the quota is maxed so you cannot download it
until its hosted elsewhere or the owner of the repo buys more quota. i am looking at [vtc1.0](https://github.com/MarvinLvn/voice-type-classifier) again for now

what i think we should use - [vtc2.0](https://github.com/LAAC-LSCP/VTC), it uses 'BabyHuBERT' instead of sincnet+lstm stack for 
better gpu parallelization and full compatability with the current cuda/pytorch stacks afaik.. so it
runs cleanly and efficiently on nvidia gpus so hope it works on the h100s 

![monkey](thinking-monkey-720p-upscale-of-480p-original-with-v0-xclffl4k6rlf1.jpg)

