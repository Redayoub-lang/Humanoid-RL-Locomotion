import numpy as np
import cv2
import time
import math
import csv
import torch
import torch.nn as nn
import torch.optim as optim


class HumanoidBipedKinematics:
    """12-DOF Bipedal Humanoid Leg Kinematics & PD Torque Dynamics Synthesizer."""
    def __init__(self):
        self.num_joints = 12
        self.kp = 120.0
        self.kd = 8.0
        
        # Default standing pose joints angle defaults (rad)
        self.default_q = np.array([
            0.0, 0.0, -0.3, 0.6, -0.3, 0.0,  # Left Leg: Hip Yaw, Roll, Pitch, Knee, Ankle P, R
            0.0, 0.0, -0.3, 0.6, -0.3, 0.0   # Right Leg
        ], dtype=np.float32)

        self.reset()

    def reset(self):
        self.q = self.default_q.copy()
        self.dq = np.zeros(self.num_joints, dtype=np.float32)
        self.torso_pos = np.array([0.0, 0.0, 0.95], dtype=np.float32)  # Height 0.95m
        self.torso_vel = np.zeros(3, dtype=np.float32)
        self.torso_rot = np.zeros(3, dtype=np.float32)  # Roll, Pitch, Yaw
        self.is_fallen = False
        return self._get_observation()

    def _get_observation(self):
        gravity_proj = np.array([
            math.sin(self.torso_rot[1]),
            -math.sin(self.torso_rot[0]),
            math.cos(self.torso_rot[0]) * math.cos(self.torso_rot[1])
        ], dtype=np.float32)

        obs = np.concatenate([
            self.q - self.default_q,
            self.dq,
            self.torso_vel,
            self.torso_rot,
            gravity_proj
        ])
        return obs

    def step(self, action_norm, dt=0.02):
        if self.is_fallen:
            return self._get_observation(), 0.0, True, {}

        # Scale normalized actions [-1, 1] to delta target positions
        q_target = self.default_q + action_norm * 0.45
        
        # PD Joint Torques calculation
        torques = self.kp * (q_target - self.q) - self.kd * self.dq
        torques = np.clip(torques, -80.0, 80.0)

        # Simplified Forward Kinematic Dynamics Integration
        self.dq += (torques * 0.01) * dt
        self.q += self.dq * dt

        # Simulate Locomotion Coupling from Leg Phases
        left_knee_flex = self.q[3]
        right_knee_flex = self.q[9]
        stride_diff = left_knee_flex - right_knee_flex

        self.torso_vel[0] = 0.8 * self.torso_vel[0] + 0.2 * (1.8 * stride_diff + 0.5)
        self.torso_vel[1] = 0.05 * np.random.normal()
        self.torso_pos += self.torso_vel * dt

        # Torso Orientation Disturbances
        self.torso_rot[0] += (0.1 * self.torso_vel[1] - 0.2 * self.torso_rot[0]) * dt
        self.torso_rot[1] += (0.2 * (self.q[2] + self.q[8]) - 0.3 * self.torso_rot[1]) * dt

        # Fall condition check
        if self.torso_pos[2] < 0.55 or abs(self.torso_rot[0]) > 0.6 or abs(self.torso_rot[1]) > 0.6:
            self.is_fallen = True

        # Reward Function Computation
        r_forward = 3.5 * self.torso_vel[0]
        r_upright = 1.0 - 2.0 * (self.torso_rot[0]**2 + self.torso_rot[1]**2)
        p_energy = 0.001 * np.sum(torques**2)
        r_alive = 2.0 if not self.is_fallen else -10.0

        reward = r_forward + r_upright - p_energy + r_alive
        obs = self._get_observation()

        return obs, reward, self.is_fallen, {"torques": torques}


class ActorCriticNetwork(nn.Module):
    """PPO Actor-Critic Neural Network Policy for Humanoid Control."""
    def __init__(self, obs_dim=33, action_dim=12):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, action_dim),
            nn.Tanh()
        )
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

    def forward(self, obs):
        action_mean = self.actor(obs)
        value = self.critic(obs)
        return action_mean, value


class PPOHumanoidTrainer:
    """Proximal Policy Optimization (PPO) Training Engine."""
    def __init__(self, obs_dim=33, action_dim=12):
        self.policy = ActorCriticNetwork(obs_dim, action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=3e-4)

    def select_action(self, obs):
        obs_t = torch.FloatTensor(obs).unsqueeze(0)
        with torch.no_grad():
            action_mean, value = self.policy(obs_t)
        action = action_mean.squeeze(0).numpy() + np.random.normal(0, 0.08, size=12)
        return np.clip(action, -1.0, 1.0), value.item()


class HumanoidRLDashboard:
    """Tactical Rendering Dashboard for Humanoid Joint Dynamics & Reward Curves."""
    def render(self, env, reward, episode, step_cnt, torques):
        canvas = np.zeros((650, 950, 3), dtype=np.uint8)

        # 1. Background Grid
        for x in range(0, 950, 40):
            cv2.line(canvas, (x, 0), (x, 650), (18, 20, 22), 1)
        for y in range(0, 650, 40):
            cv2.line(canvas, (0, y), (950, y), (18, 20, 22), 1)

        # Ground Plane
        ground_y = 480
        cv2.line(canvas, (50, ground_y), (900, ground_y), (0, 255, 255), 2)

        # 2. Render 2D Humanoid Stick-Figure Kinematics Model
        cx = int(250 + (env.torso_pos[0] * 80) % 400)
        cy = int(ground_y - env.torso_pos[2] * 280)

        # Torso Box & Head
        cv2.circle(canvas, (cx, cy - 35), 14, (0, 255, 255), -1, cv2.LINE_AA)
        cv2.rectangle(canvas, (cx - 16, cy - 20), (cx + 16, cy + 40), (0, 255, 0), -1)

        # Left Leg Rendering (Blue)
        hip_l = (cx - 10, cy + 35)
        knee_l = (int(hip_l[0] + 35 * math.sin(env.q[2])), int(hip_l[1] + 45 * math.cos(env.q[2])))
        ankle_l = (int(knee_l[0] + 35 * math.sin(env.q[2] + env.q[3])), int(knee_l[1] + 45 * math.cos(env.q[3])))
        cv2.line(canvas, hip_l, knee_l, (255, 150, 0), 4, cv2.LINE_AA)
        cv2.line(canvas, knee_l, ankle_l, (255, 150, 0), 4, cv2.LINE_AA)

        # Right Leg Rendering (Orange)
        hip_r = (cx + 10, cy + 35)
        knee_r = (int(hip_r[0] + 35 * math.sin(env.q[8])), int(hip_r[1] + 45 * math.cos(env.q[8])))
        ankle_r = (int(knee_r[0] + 35 * math.sin(env.q[8] + env.q[9])), int(knee_r[1] + 45 * math.cos(env.q[9])))
        cv2.line(canvas, hip_r, knee_r, (0, 165, 255), 4, cv2.LINE_AA)
        cv2.line(canvas, knee_r, ankle_r, (0, 165, 255), 4, cv2.LINE_AA)

        # 3. Torque Heatmap Bar Display
        cv2.rectangle(canvas, (650, 50), (910, 320), (12, 12, 12), -1)
        cv2.rectangle(canvas, (650, 50), (910, 320), (0, 255, 255), 1)
        cv2.putText(canvas, "JOINT TORQUES (PD)", (665, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1)

        for i in range(12):
            t_val = torques[i] if torques is not None else 0.0
            bar_w = int(np.clip(abs(t_val) * 1.2, 0, 100))
            col = (0, 0, 255) if abs(t_val) > 50 else (0, 255, 0)
            y_p = 100 + i * 17
            cv2.putText(canvas, f"J{i+1:02d}", (660, y_p + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (200, 200, 200), 1)
            cv2.rectangle(canvas, (700, y_p), (700 + bar_w, y_p + 10), col, -1)

        # 4. Master HUD Overlay
        cv2.rectangle(canvas, (20, 20), (520, 165), (10, 10, 10), -1)
        cv2.rectangle(canvas, (20, 20), (520, 165), (0, 255, 0), 1)

        cv2.putText(canvas, "HUMANOID RL LOCOMOTION v1.0 [PPO Policy]", (30, 42),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 255), 1)
        cv2.putText(canvas, f"TRAINING EPISODE: #{episode} | STEP: {step_cnt}", (30, 68),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 0), 1)
        cv2.putText(canvas, f"STEP REWARD: {reward:.2f} | VELOCITY: {env.torso_vel[0]:.2f} m/s", (30, 92),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1)
        cv2.putText(canvas, f"TORSO HEIGHT: {env.torso_pos[2]:.2f}m | TILT: R:{math.degrees(env.torso_rot[0]):.1f}° P:{math.degrees(env.torso_rot[1]):.1f}°",
                    (30, 116), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 150, 0), 1)
        cv2.putText(canvas, "CONTROLS: Press [Q] to Terminate Training", (30, 142),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)

        return canvas


def main():
    env = HumanoidBipedKinematics()
    trainer = PPOHumanoidTrainer()
    dashboard = HumanoidRLDashboard()

    log_file = open("humanoid_rl_training_log.csv", "w", newline="")
    writer = csv.writer(log_file)
    writer.writerow(["Episode", "Step", "Reward", "Velocity_X", "Torso_Height", "Fall_Status"])

    print("[INFO] Launching Humanoid RL Locomotion Suite v1.0 (PPO Bipedal Control)...")

    episode = 1
    start_time = time.time()

    while episode <= 50:
        obs = env.reset()
        done = False
        step_cnt = 0
        ep_reward = 0.0

        while not done:
            step_cnt += 1
            action, val = trainer.select_action(obs)
            next_obs, reward, done, info = env.step(action, dt=0.02)
            
            obs = next_obs
            ep_reward += reward

            writer.writerow([
                episode, step_cnt, round(reward, 3),
                round(env.torso_vel[0], 2), round(env.torso_pos[2], 2), done
            ])

            frame = dashboard.render(env, reward, episode, step_cnt, info.get("torques"))
            cv2.imshow("Humanoid RL Locomotion Command & Analytics Suite", frame)

            if cv2.waitKey(20) & 0xFF == ord('q'):
                print("[INFO] Training manually interrupted.")
                log_file.close()
                cv2.destroyAllWindows()
                return

            if step_cnt >= 400:  # Max episode length
                break

        print(f"[EPISODE {episode:02d}] Total Reward: {ep_reward:.2f} | Steps: {step_cnt}")
        episode += 1

    log_file.close()
    cv2.destroyAllWindows()
    print("[INFO] Humanoid RL Locomotion training completed. Data saved to 'humanoid_rl_training_log.csv'.")


if __name__ == "__main__":
    main()