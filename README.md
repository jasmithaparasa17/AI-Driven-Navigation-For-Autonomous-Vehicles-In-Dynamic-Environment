```markdown
# 🚗 AI-Driven Navigation for Autonomous Vehicles in Dynamic Environment

This project demonstrates an AI-powered autonomous vehicle trained to navigate through dynamic environments using Deep Q-Learning (DQN). Built using the CarRacing-v3 environment from OpenAI Gym, the model is trained to learn optimal driving strategies. A Streamlit app is also included for real-time performance visualization and interaction.

---

## 📌 Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [Setup Instructions](#setup-instructions)
- [How to Train the Model](#how-to-train-the-model)
- [How to Run the Streamlit App](#how-to-run-the-streamlit-app)
- [Model Details](#model-details)
- [Outputs](#outputs)
- [Dependencies](#dependencies)
- [License](#license)

---

## ✨ Features

- ✅ Deep Q-Network (DQN) based reinforcement learning
- ✅ Uses OpenAI Gym’s CarRacing-v3 simulation
- ✅ Learns adaptive navigation in dynamic environments
- ✅ Real-time performance visualization with Streamlit
- ✅ Modular code structure for training, evaluation, and deployment

---

## 📁 Project Structure



├── app.py                  # Streamlit dashboard for visualization
├── dqn\_model.py            # DQN architecture definition
├── training\_dataset.py     # Model training script
├── dqn\_carracing\_model.pth # Trained model weights
├── README.md               # Project documentation
├── LICENSE                 # License info
└── requirements.txt        # Python dependencies
```
````

---

## ⚙️ Setup Instructions

Follow the steps below to set up the environment:

1. **Clone the Repository**

   ```bash
   git clone https://github.com/yourusername/ai-driven-navigation.git
   cd ai-driven-navigation
````

2. **Create a Virtual Environment (Optional but Recommended)**

   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Install Additional Packages if Required**

   ```bash
   pip install gym[box2d] streamlit matplotlib torch torchvision
   ```

---

## 🏁 How to Train the Model

> Note: Make sure you have a compatible GPU or a high-performing CPU. Training time may vary based on system specifications.

1. **Train the Model**

   ```bash
   python training_dataset.py
   ```

2. The script will:

   * Initialize the DQN agent.
   * Interact with the CarRacing-v3 environment.
   * Collect rewards and optimize the policy.
   * Save trained weights to `dqn_carracing_model.pth`.

---

## 📊 How to Run the Streamlit App

Once the model is trained, launch the dashboard:

```bash
streamlit run app.py
```

This opens a browser-based UI where you can:

* Load the trained model
* View real-time navigation
* Visualize reward and performance plots

---

## 🧠 Model Details

* **Input**: Visual frames from CarRacing-v3 environment
* **Output**: Continuous action values (steering, throttle, brake)
* **Model Type**: Deep Q-Network (DQN)
* **Objective**: Maximize cumulative reward over each episode

---

## 📈 Outputs

* ✅ `dqn_carracing_model.pth`: Trained model weights
* ✅ Real-time interactive driving dashboard (via Streamlit)
* ✅ Reward progression plots
* ✅ Performance comparison charts (if implemented)

---

## 📦 Dependencies

All dependencies are listed in `requirements.txt`. Key libraries include:

* `gym[box2d]`
* `streamlit`
* `numpy`
* `matplotlib`
* `torch`
* `torchvision`
* `opencv-python`

Install them using:

```bash
pip install -r requirements.txt
```

---

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for details.

---

## 🙋‍♀️ Contact

For any queries or feedback, feel free to reach out at:
📧 [jasmithaparasa17@gmail.com](mailto:jasmithaparasa17@gmail.com)
🔗 [LinkedIn](https://www.linkedin.com/in/jasmitha-parasa17/)


