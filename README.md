# Instructions
1. Clone repo
2. Clone https://huggingface.co/KhaldiAbderrhmane/resnet50-facial-emotion-recognition
3. Rename "resnet50-facial-emotion-recognition" to "facialrecognition_resnet50"
4. Install the neccessary requirements
> Python 3.12.9

> FYI I use apple silicon accelleration for torch, which is the nightly version, probably different for CUDA

> For rtmidi you may have to use rtmidi-python

> Best to go step by step through the notebook

5. Update dependencies and vars such as camera_id inside autoparam.py. CLI will guide you through mapping the parameters inside Ableton Live.

> So far I've not tested saving and loading the model

Have fun!

________
> Accuracy for face emotion detection dramatically increased when close to camera, probably face detection and auto crop would be neccessary for practical application