# SpeechLab
the container files/the container are located in [`speechlab_diarization`](/speechlab_diarization)
... still working on testing it

![monkey](thinking-monkey-720p-upscale-of-480p-original-with-v0-xclffl4k6rlf1.jpg)

link to what im using in the container - [vtc2.0](https://github.com/LAAC-LSCP/VTC), it uses 'BabyHuBERT' instead of sincnet+lstm stack for 
better gpu parallelization and full compatability with the current cuda/pytorch stacks afaik.. so it
runs cleanly and efficiently on nvidia gpus so hope it works on the h100s 