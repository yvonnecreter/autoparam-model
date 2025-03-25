# PARAMETERS
camera_id = 0
n_midi_params = int(input("Enter the number of MIDI parameters: "))

# IMPORTS
import cv2
import numpy as np
import cv2
import numpy as np
import torch
import torchvision.models as models
from torchvision import transforms
import matplotlib.pyplot as plt
from safetensors.torch import load_file
import numpy as np
import time
import gym
from gym import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import mido


# IMAGE CAPTURE
cap = cv2.VideoCapture(camera_id)
def shoot_image():
    """Capture an image from the camera and return it as a numpy array."""
    ret, frame = cap.read()
    if not ret:
        return None
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return rgb_frame

# PREPROCESSING
def crop_and_resize(image, target_size=(224, 224)):
    """Crop and resize the image to the target size."""
    # Get dimensions of the image
    h, w = image.shape[:2]
    
    # Center crop to make it square
    if h > w:
        start = (h - w) // 2
        end = start + w
        cropped = image[start:end, :]
    else:
        start = (w - h) // 2
        end = start + h
        cropped = image[:, start:end]
    
    # Resize to target size
    resized = cv2.resize(cropped, target_size)
    return resized

# Preprocess image
def process_frame(x):
    """Preprocess the image for ResNet50."""
    x = transforms.ToTensor()(x)
    x = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(x)
    x = x.unsqueeze(0)
    return x

img = shoot_image()
img_cropped_resized = crop_and_resize(img)
x = process_frame(img_cropped_resized)

# MODEL
# EMOTION DETECTION
img = shoot_image()
img = process_frame(crop_and_resize(img))

resnet = models.resnet50(pretrained=False)
resnet.fc = torch.nn.Linear(resnet.fc.in_features, 6)

safetensors_path = "facialrecognition_resnet50/model.safetensors"  # Replace with your file path
state_dict = load_file(safetensors_path)
state_dict = {k.replace("resnet.", ""): v for k, v in state_dict.items()}
resnet.load_state_dict(state_dict=state_dict)
resnet.eval()

labels = ["Surprise", "Angry", "Fear", "Sad", "Happy", "?"]
labels_map = {"Surprise": 0, "Angry": 1, "Fear": 2, "Sad": 3, "Happy": 4, "?": 5}
def get_sentiment_score(emotion_selection='Happy'):
    """
    Get Sentiment score.
    """
    img = process_frame(crop_and_resize(shoot_image()))
    y = resnet(img)
    probs = torch.nn.functional.softmax(y, dim=1)
    return torch.topk(probs, k=5)[0][0][labels_map[emotion_selection]]

# MIDI

## init outputs
try:
    midi_out = mido.open_output('Virtual MIDI Port', virtual=True)  # Create a virtual MIDI port
    print("MIDI output port opened successfully.")
except Exception as e:
    print(f"Failed to open MIDI output port: {e}")
    midi_out = None

midi_params = np.random.randint(0, 128, size=n_midi_params)

def send_midi_data(params):
    """
    Send MIDI control change messages for each parameter.
    """
    if midi_out is not None:
        for i, value in enumerate(params):
            # Send control change message (channel 0, control number i, value)
            msg = mido.Message('control_change', control=i, value=int(value))
            midi_out.send(msg)
            # print(f"Sent MIDI message: {msg}")

def smooth_transition(start_params, end_params, ms_length=2000):
    """
    Smoothly transition from start_params to end_params over ms_length milliseconds.
    """
    steps = 100  # Number of interpolation steps
    delay = ms_length / steps  # Time delay between steps

    for i in range(steps):
        # Linear interpolation
        current_params = start_params + (end_params - start_params) * (i / steps)
        current_params = np.clip(current_params, 0, 127)  # Clip to valid MIDI range
        send_midi_data(current_params)
        time.sleep(delay / 1000)  # Convert ms to seconds

# MAP SIGNAL
input('Prepare for MIDI Mappping. Select Param1 then press enter.')
for i in range(n_midi_params):
    # TEST SIGNAL
    for _ in range(30):
        msg = mido.Message('control_change', control=i, value=int(42))
        midi_out.send(msg)
        time.sleep(0.1)
    print('Done for signal', i)
    input('Press enter to continue, once you have selected next Parameter')
print("Done!")

# REINFORCEMENT LEARNING
## Gym Environment
class MidiSentimentEnv(gym.Env):
    def __init__(self):
        super(MidiSentimentEnv, self).__init__()
        # Define action and observation space
        self.action_space = spaces.Box(low=-10, high=10, shape=(n_midi_params,), dtype=np.float32)  # Interval changes
        self.observation_space = spaces.Box(low=0, high=127, shape=(n_midi_params,), dtype=np.float32)  # MIDI parameters
        self.i = 0
        # Initialize state
        self.state = np.random.randint(0, 128, size=n_midi_params)
        self.ms_length = 2000  # Initial time window
    
    def seed(self, seed=None):
        np.random.seed(seed)

    def reset(self):
        # Reset the environment to an initial state
        self.state = np.random.randint(0, 128, size=n_midi_params)
        return self.state

    def step(self, action):
        print(self.i)
        self.i += 1
        action = action.astype(int)  # Convert action to integer
        self.state += action  # Now self.state remains int64
        self.state = np.clip(self.state, 0, 127)  # Clip to valid MIDI range

        # send_midi_data(self.state)
        target_params = np.random.randint(0, 128, size=n_midi_params)
        smooth_transition(midi_params, target_params)
        
        sentiment_score = get_sentiment_score()
        reward = sentiment_score
        done = False  # Infinite episode

        return self.state, reward, done, {}

    def render(self, mode='human'):
        # Optional: Visualize the environment
        print("Current MIDI Parameters:", self.state)

input("Press enter to start training")
# Create the environment
env = make_vec_env(MidiSentimentEnv, n_envs=1)
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000)

# # Save the model
model.save("midi_sentiment_rl_model")
model = PPO.load("midi_sentiment_rl_model")