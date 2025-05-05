import streamlit as st
import gymnasium as gym
import numpy as np
import torch
from dqn_model import DQN
from PIL import Image
import time


# Example values – update these based on your environment!
state_dim = 96 * 96 * 3  # Flattened image size
action_dim = 4           # Number of discrete actions

model = DQN(state_dim=27648, action_dim=3)  # match saved model


# Load model


state_dict = torch.load(r"C:\Users\jasmi\OneDrive\Desktop\Project Codes\dqn\dqn\dqn_ep2.pth", map_location=torch.device('cpu'))

model.load_state_dict(state_dict)
model.eval()

# Streamlit UI
st.set_page_config(page_title="CarRacing AI", layout="centered")
st.title("🚘 AI-Driven Car Racing Navigation")

run = st.button("Start Racing")

if run:
    env = gym.make("CarRacing-v3", render_mode="rgb_array")
    obs, _ = env.reset()
    obs = obs.flatten() / 255.0
    done = False
    total_reward = 0

    # Create a placeholder for updating frames
    frame_placeholder = st.empty()

    while not done:
        with torch.no_grad():
            action = model(torch.FloatTensor(obs)).argmax().item()

        # Map discrete action to continuous
        action_array = np.array([0.0, 0.0, 0.0])
        if action == 0:
            action_array[0] = -1.0; action_array[1] = 0.5
        elif action == 1:
            action_array[1] = 1.0
        elif action == 2:
            action_array[0] = 1.0; action_array[1] = 0.5

        next_obs, reward, done, _, _ = env.step(action_array)
        obs = next_obs.flatten() / 255.0
        total_reward += reward

        # Render and update the frame
        frame = env.render()
        img = Image.fromarray(frame)
        frame_placeholder.image(img, caption=f"🎯 Total Reward: {int(total_reward)}", use_column_width=True)

        time.sleep(0.05)  # Slow down rendering slightly

    st.success(f"🏁 Finished! Final Reward: {int(total_reward)}")
