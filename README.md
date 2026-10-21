# Humanoid-RL-Locomotion (v1.0 Next-Gen Edition)
## Deep Reinforcement Learning for 12-DOF Bipedal Locomotion via Proximal Policy Optimization

An advanced robotics suite for training stable bipedal walking and dynamic balance policies on a 12-DOF Humanoid Robot using **Proximal Policy Optimization (PPO)** and PD torque controller synthesis.

---

## Technical Features

* **12-DOF Kinematic Model:** Simulates double-leg joint architectures (Hip Yaw/Roll/Pitch, Knee Pitch, Ankle Pitch/Roll).
* **PPO Actor-Critic Policy:** Tanh-activated neural networks mapping 33-dim continuous observations to normalized PD joint target references.
* **PD Joint Torque Controller:** Low-level torque generation ($\boldsymbol{\tau} = K_p (\mathbf{q}_{\text{ref}} - \mathbf{q}) - K_d \dot{\mathbf{q}}$) bounded under actuator saturation limits.
* **Real-time Locomotion Dashboard:** Live visualization of stick-figure joint linkage, torque heatmaps, orientation roll/pitch angles, and step rewards.
* **Training Metrics Exporter:** Logged episode step metrics saved to `humanoid_rl_training_log.csv`.

---

## Mathematical Formulation

PPO Clipped Objective Function:

$$L^{\text{CLIP}}(\theta) = \hat{\mathbb{E}}_t \left[ \min\left( r_t(\theta) \hat{A}_t, \, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t \right) \right]$$

Multi-Objective Reward Function:

$$R_t = w_v \cdot v_x - w_y \cdot \vert{}v_y\vert{} - w_{\text{orient}} \cdot \Vert{}\mathbf{z}_{\text{torso}} - \mathbf{e}_z\Vert{}^2 - w_{\tau} \cdot \Vert{}\boldsymbol{\tau}_t\Vert{}^2 + R_{\text{alive}}$$

---

## Quick Execution Guide

```bash
# Install requirements
pip install -r requirements.txt

# Run PPO Humanoid Training Suite
python humanoid_rl_engine.py