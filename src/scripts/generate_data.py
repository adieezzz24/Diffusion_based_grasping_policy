import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from omni.isaac.lab.app import AppLauncher

# Launch Isaac Sim (GUI mode to visualize data collection)
app_launcher = AppLauncher(headless=False)
simulation_app = app_launcher.app

import torch
from src.envs.ur5_grasp_env import UR5GraspEnv
from src.utils.simulation_utils import create_hdf5_storage, save_step_to_hdf5, calculate_expert_action

def main():
    # 1. Initialize Environment
    env = UR5GraspEnv(headless=False)
    
    # 2. Setup Data Storage
    save_dir = "../data/raw/demo_data.hdf5"
    # Assuming action dim is 6 (UR5 joints) and obs dim matches robot state + images (images stored separately)
    action_dim = 6 
    obs_dim = 6 
    create_hdf5_storage(save_dir, obs_dim, action_dim)
    
    NUM_EPISODES = 50 # Small number for testing, increase to 100-500 later
    MAX_STEPS = 200

    print("Starting Data Collection...")

    for episode in range(NUM_EPISODES):
        env.reset()
        
        for step in range(MAX_STEPS):
            # A. Get Observation
            image_tensor, joint_pos = env.get_observation()
            
            # Convert image to numpy uint8 for storage
            # Isaac Sim gives [0, 1] float, convert to [0, 255] uint8
            image_np = (image_tensor.cpu().numpy() * 255).astype(np.uint8)
            
            # B. Get Expert Action
            # Retrieve object position from env (simulated access)
            # cube_pos = env.cube.data.root_pos 
            action = calculate_expert_action(env.robot, cube_pos=None) 
            
            obs = joint_pos[0].cpu().numpy() # Flatten for storage
            
            # C. Save to HDF5
            terminal = (step == MAX_STEPS - 1)
            save_step_to_hdf5(save_dir, obs, action, image_np, terminal)
            
            # D. Step Simulation
            env.step()
            
            # Optional: Break if cube grasped/lifted (check logic here)
            
        print(f"Episode {episode+1}/{NUM_EPISODES} completed.")

    print("Data collection finished. Closing simulation.")
    simulation_app.close()

if __name__ == "__main__":
    main()