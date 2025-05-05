
import cv2
import torch
from ultralytics import YOLO
from google.colab.patches import cv2_imshow # Import the cv2_imshow function


# Load Pre-trained YOLO Model
model = YOLO("yolov8n.pt")  # Uses YOLOv8 small model

# Load Image
#image = cv2.imread("/content/Road_Scene.jpg")
image = cv2.imread("Road_image.jpg")
# Object Detection
results = model(image)

# Display Results
for box in results[0].boxes:
    x1, y1, x2, y2 = box.xyxy[0]
    cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

cv2_imshow(image)
cv2.waitKey(0)
cv2.destroyAllWindows()

import cv2
import numpy as np

# Initialize ORB Feature Detector
orb = cv2.ORB_create()

# Load Consecutive Frames
frame1 = cv2.imread("Frame_1.jpg", 0)
frame2 = cv2.imread("Frame_2.jpg", 0)

# Find Keypoints & Descriptors
kp1, des1 = orb.detectAndCompute(frame1, None)
kp2, des2 = orb.detectAndCompute(frame2, None)

# Match Features
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)

# Sort Matches by Distance
matches = sorted(matches, key=lambda x: x.distance)

# Draw Matches
matched_image = cv2.drawMatches(frame1, kp1, frame2, kp2, matches[:50], None)

cv2_imshow(matched_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

import heapq

class Node:
    def __init__(self, x, y, cost=0, parent=None):
        self.x, self.y, self.cost, self.parent = x, y, cost, parent

    def __lt__(self, other):
        return self.cost < other.cost

def astar(grid, start, goal):
    open_list, closed_set = [], set()
    heapq.heappush(open_list, (0, Node(*start)))

    while open_list:
        _, current = heapq.heappop(open_list)

        if (current.x, current.y) == goal:
            path = []
            while current:
                path.append((current.x, current.y))
                current = current.parent
            return path[::-1]

        closed_set.add((current.x, current.y))

        for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
            x, y = current.x + dx, current.y + dy
            if 0 <= x < len(grid) and 0 <= y < len(grid[0]) and grid[x][y] == 0 and (x,y) not in closed_set:
                heapq.heappush(open_list, (current.cost+1, Node(x, y, current.cost+1, current)))

    return []

# Example Grid (0=Free Path, 1=Obstacle)
grid = [[0, 0, 0, 1, 0],
        [0, 1, 0, 1, 0],
        [0, 0, 0, 0, 0],
        [1, 1, 0, 1, 1],
        [0, 0, 0, 0, 0]]

# Start & Goal Positions
start, goal = (0, 0), (4, 4)

# Compute Optimal Path
path = astar(grid, start, goal)
print("Optimal Path:", path)

# Example Grid (0=Free Path, 1=Obstacle)
grid = [[0, 0, 0, 1, 0],
        [0, 1, 0, 1, 0],
        [0, 0, 0, 0, 0],
        [1, 1, 0, 1, 1],
        [0, 0, 0, 0, 0]]

# Start & Goal Positions
start, goal = (0, 0), (4, 4)

# Compute Optimal Path
path = astar(grid, start, goal)
print("Optimal Path:", path)









!apt-get install swig -y
!pip install gymnasium[box2d] pygame imageio

!pip install gymnasium[box2d] pygame


!sudo apt-get install -y swig
!pip install gymnasium[box2d] box2d-py pygame


import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import matplotlib.pyplot as plt

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the DQN Network
class DQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(DQN, self).__init__()
        c, h, w = input_shape
        # Define self.net before using it
        self.net = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),

        )
        # Calculate feature size after net definition
        self.feature_size_value = self.feature_size(input_shape)

        # Continue with the rest of the network
        self.net = nn.Sequential(
            *self.net, # unpack previous layers
            nn.Linear(self.feature_size_value, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )


    def feature_size(self, shape):
        # Create a temporary sequential model to calculate the feature size
        temp_net = nn.Sequential(
            nn.Conv2d(shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        return temp_net(torch.zeros(1, *shape)).view(1, -1).size(1)

    def forward(self, x):
        return self.net(x)

# ... (rest of the code remains the same) ...
# Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size=32):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.array, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)

def preprocess(obs):
    # obs is already in the format (96, 96, 3), don't need to index
    # obs = obs[0]  # remove (obs, info) - This line is the problem
    obs = np.mean(obs, axis=2, keepdims=True)  # grayscale
    obs = obs.transpose((2, 0, 1)) / 255.0  # to CHW and normalize
    return obs.astype(np.float32)

# Hyperparameters
env = gym.make("CarRacing-v3", render_mode=None)
obs_shape = (1, 96, 96)
# Get number of actions from action space shape
# n_actions = env.action_space.shape[0]
# Since the action space is continuous, we need to discretize it
n_actions = 9  # Example: 3 steering angles * 3 acceleration/brake combinations


# Define action space discretization
def discretize_action(action):
    # Example discretization:
    # Steering: [-1, 0, 1]
    # Gas/Brake: [0, 0.5, 1]
    steering = np.clip(action[0], -1, 1)
    gas = np.clip(action[1], 0, 1)
    brake = np.clip(action[2], 0, 1)

    steering_idx = int(np.round((steering + 1) / 2 * 2))  # Map to 0, 1, 2
    gas_idx = int(np.round(gas * 2))  # Map to 0, 1, 2
    brake_idx = int(np.round(brake * 2))  # Map to 0, 1, 2

    # Combine indices into a single action index
    # Adjust the formula based on your desired discretization
    action_idx = steering_idx * 3 + gas_idx

    return action_idx


model = DQN(obs_shape, n_actions).to(device)
target_model = DQN(obs_shape, n_actions).to(device)
target_model.load_state_dict(model.state_dict())

optimizer = optim.Adam(model.parameters(), lr=1e-4)
buffer = ReplayBuffer()
epsilon = 1.0
gamma = 0.99
episodes = 3
batch_size = 32
sync_every = 10

for ep in range(episodes):
    obs, _ = env.reset()
    state = preprocess(obs)
    episode_reward = 0
    done = False

    while not done:
        if random.random() < epsilon:
            action = env.action_space.sample()
            # Discretize the action for the DQN
            action_idx = discretize_action(action)
        else:
            with torch.no_grad():
                state_tensor = torch.tensor(state).unsqueeze(0).to(device)
                q_values = model(state_tensor)
                # Get the action index with the highest Q-value
                action_idx = torch.argmax(q_values).item()
                # Convert the discretized action index back to continuous action
                # ... (Implement the reverse mapping from action_idx to continuous action)

        next_obs, reward, terminated, truncated, _ = env.step(action) # Use the selected action
        done = terminated or truncated
        next_state = preprocess(next_obs)

        # Store the discretized action index in the buffer
        buffer.push(state, action_idx, reward, next_state, done)
        state = next_state
        episode_reward += reward

        # Learn
        if len(buffer) >= batch_size:
            s, a, r, s_, d = buffer.sample(batch_size)
            s = torch.tensor(s).to(device)
            # Ensure action 'a' is of type long (int64)
            a = torch.tensor(a, dtype=torch.int64).to(device)
            r = torch.tensor(r).to(device)
            s_ = torch.tensor(s_).to(device)
            d = torch.tensor(d).to(device)

            q_vals = model(s).gather(1, a.unsqueeze(1)).squeeze(1)
            with torch.no_grad():
                next_q_vals = target_model(s_).max(1)[0]
            target = r.type(torch.float32) + gamma * next_q_vals * (1 - d.type(torch.float32)) # Change r to r.type(torch.float32)
            loss = nn.MSELoss()(q_vals, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Epsilon decay
    epsilon = max(0.1, epsilon * 0.995)

    # Sync target model
    if ep % sync_every == 0:
        target_model.load_state_dict(model.state_dict())

    print(f"Episode {ep+1}: Reward = {episode_reward:.2f}, Epsilon = {epsilon:.2f}")

print("Training complete!")

import torch

# Save trained model
model_path = "/content/dqn_carracing_model.pth"
torch.save(model.state_dict(), model_path)
print(f"Model saved to {model_path}")

# Verify installation
import torch
print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

!pip install streamlit


import streamlit as st
import gymnasium as gym
import torch
import numpy as np
from PIL import Image
import tempfile
import os
import imageio
import shutil

# Load the trained model
class DQN(torch.nn.Module):
    def __init__(self, input_shape, n_actions):
        super(DQN, self).__init__()
        c, h, w = input_shape
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(c, 32, kernel_size=8, stride=4),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, kernel_size=4, stride=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, kernel_size=3, stride=1),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
            torch.nn.Linear(self.feature_size(input_shape), 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, n_actions)
        )

    def feature_size(self, shape):
        return self.net[:4](torch.zeros(1, *shape)).view(1, -1).size(1)

    def forward(self, x):
        return self.net(x)

# Preprocess the observation
def preprocess(obs):
    obs = np.mean(obs, axis=2, keepdims=True)  # grayscale
    obs = obs.transpose((2, 0, 1)) / 255.0
    return obs.astype(np.float32)

# Load model from file
@st.cache_resource
def load_model(model_path, obs_shape, n_actions):
    model = DQN(obs_shape, n_actions)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

# Generate gameplay frames
def generate_episode(model, env):
    frames = []
    obs, _ = env.reset()
    state = preprocess(obs)
    done = False

    while not done:
        with torch.no_grad():
            state_tensor = torch.tensor(state).unsqueeze(0)
            q_values = model(state_tensor)
            action = torch.argmax(q_values).item()

        obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        frames.append(Image.fromarray(obs))
        state = preprocess(obs)

# Save GIF inside this function
    gif_path = "dqn_simulation.gif"
    imageio.mimsave(gif_path, frames, fps=5)
    return gif_path  # return the GIF path instead of frames

# Main Streamlit UI
def main():
    st.title("🎮 DQN Car Racing - Demo")
    st.write("Play a trained DQN model on the `CarRacing-v3` environment.")

    model_path = st.file_uploader("", type=["pth"])
    if model_path:
        obs_shape = (1, 96, 96)
        n_actions = 5  # Steering + Gas + Brake options

        st.success("Model uploaded successfully.")
        model = load_model(model_path, obs_shape, n_actions)
        env = gym.make("CarRacing-v3", render_mode="rgb_array")

        st.info("Generating simulation...")
        frames = generate_episode(model, env)

        # Save frames as a temporary GIF
        tmp_dir = tempfile.mkdtemp()
        gif_path = os.path.join(tmp_dir, "episode.gif")
        imageio.mimsave(gif_path, frames, fps=30)

        st.image(gif_path, caption="DQN Agent Playing", use_column_width=True)
        shutil.rmtree(tmp_dir)

if __name__ == "__main__":
    main()





