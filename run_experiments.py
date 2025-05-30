import os
import csv
import pickle
import json
import gymnasium as gym
import cv2
import numpy as np
from tqdm import tqdm
from gymnasium.wrappers import RecordVideo, OrderEnforcing
from llms import Agent

# Custom wrapper to prevent video closing on reset
class ContinuousRecordVideo(RecordVideo):
    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        self.video_recorder.capture_frame()
        return observation, reward, terminated, truncated, info

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        if self.video_recorder:
            self.video_recorder.capture_frame()
        return observation, info


class Run:
    def __init__(self, env_name, provider, output_dir="./experiments/", prompt=None, model_id=None):
        self.provider = provider.lower()  # 'openai', 'gemini', 'claude' or 'rand'
        self.sys_prompt = prompt or ""
        self.env_name = env_name
        self.action_list = []
        self.cum_rewards = []
        self.rewards = 0
        self.steps_taken = 0
        self.num_timesteps = 1000

        # Simplify ALE names
        self.temp_env_name = env_name.replace("ALE/", "")
        base = self.temp_env_name[:-3]
        self.new_dir = os.path.join(output_dir, f"{base}_{provider}/")
        os.makedirs(self.new_dir, exist_ok=True)

        # Skip test if already done
        state_file = os.path.join(self.new_dir, f"env_{base}_state.pkl")
        if os.path.exists(state_file):
            print(f"\n\nEnvironment '{env_name}' already processed—skipping.\n\n")
            return

        # Build environment and video recorder
        env = gym.make(env_name, render_mode="rgb_array")
        env = OrderEnforcing(env, disable_render_order_enforcing=True)
        env.reset()
        self.env = ContinuousRecordVideo(env=env,
                                         video_folder=self.new_dir,
                                         name_prefix=base + "_rollout")

        # mapping of model IDs by provider
        if self.provider == 'rand':
            self.rand_rollout()
        else:
            if model_id == None:
                self.MODELS = {
                    "openai":  ["gpt-4-turbo", "gpt-4.1-mini"], 
                    "claude":  ["claude-3-opus-20240229","claude-3-opus-20240229"], 
                    "gemini":  ["gemini-2.5-flash-preview-04-17","gemini-2.5-flash-preview-04-17"],
                }

                model_id = self.MODELS[self.provider][1]
                self.model = Agent(
                    provider=self.provider,
                    model_name=model_id,
                    system_prompt=self.sys_prompt,
                    env=self.env
                )

            else: 
                self.model = Agent(
                    provider=self.provider,
                    model_name=model_id,
                    system_prompt=self.sys_prompt,
                    env=self.env
                )
            self.model_rollout()

        # Write summary CSV
        with open(os.path.join(self.new_dir, "actions_rewards.csv"), "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["action", "cumulative_reward"])
            for a, r in zip(self.action_list, self.cum_rewards):
                writer.writerow([a, r])


    def save_states(self, rewards, action):
        state = self.env.env.ale.cloneState()
        rand_state = getattr(self.env.env, 'np_random', self.env.env.unwrapped.np_random)
        self.states = getattr(self, 'states', [])
        self.states.append((state, rand_state, rewards, self.steps_taken, action))
        with open(os.path.join(self.new_dir, f"env_{self.temp_env_name[:-3]}_state.pkl"), "wb") as f:
            pickle.dump(self.states, f)


    def rand_rollout(self):
        self.env.start_video_recorder()
        obs, info = self.env.reset()
        self.save_states(self.rewards, 0)
        pbar = tqdm(total=self.num_timesteps, desc=f"Random Rollout ({self.temp_env_name})", unit="step")

        for _ in range(self.num_timesteps - self.steps_taken):
            obs = cv2.resize(obs, (512, 512))
            action = self.env.action_space.sample()
            self.action_list.append(action)
            obs, rew, term, trunc, info = self.env.step(action)
            self.save_states(self.rewards, action)
            self.rewards += rew
            self.cum_rewards.append(self.rewards)
            if term or trunc:
                obs, info = self.env.reset()
            self.steps_taken += 1
            pbar.update(1)
            pbar.set_postfix({"reward": self.rewards})

        pbar.close()
        print(f"Total reward (random): {self.rewards}")
        self.env.close_video_recorder()
        self.env.close()


    def model_rollout(self):
        prompt = (
            'Analyze this game frame and select the optimal action. '
            'Respond with JSON: {"reasoning":"<text>","action":<int>}.'
        )

        self.env.start_video_recorder()
        obs, info = self.env.reset()
        self.save_states(self.rewards, 0)
        pbar = tqdm(total=self.num_timesteps, desc=f"{self.provider} Rollout ({self.temp_env_name})", unit="step")

        for n in range(self.num_timesteps - self.steps_taken):
            obs = cv2.resize(obs, (512, 512))

            if n < 15 or n % 2 == 0:
                # No-op for initial frames and evens
                action = 0
            else:
                # Send frame + prompt
                self.model.add_user_message(text=prompt, frame=obs)

                # Get model answer
                action, reasoning = self.model.generate_response(self.new_dir)
                # Put assistant message into context
                self.model.add_assistant_message(text=reasoning)

            # Record action & step
            self.action_list.append(action)
            obs, rew, term, trunc, info = self.env.step(action)
            self.env.render()
            self.save_states(self.rewards, action)
            self.rewards += rew
            self.cum_rewards.append(self.rewards)
            
            # Context buffer of only the 4 most recent frames
            # Delete oldest context
            self.model.delete_messages()

            if term or trunc:
                obs, info = self.env.reset()

            self.steps_taken += 1
            pbar.update(1)
            pbar.set_postfix({"reward": self.rewards})

        pbar.close()
        self.env.close_video_recorder()
        self.env.close()
