from dm_control import mujoco, mjcf
import numpy as np
import time

def create_limb(length, thickness, rgba):
    model = mjcf.RootElement()
    model.default.joint.damping = 2
    model.default.joint.type = 'hinge'
    model.default.geom.type = 'capsule'
    model.default.geom.rgba = rgba
    limb = model.worldbody.add('body')
    limb_joint = limb.add('joint', axis=[0, 0, 1])
    limb.add('geom', fromto=[0, 0, 0, length, 0, 0], size=[thickness, length])
    model.actuator.add('position', joint=limb_joint, kp=10)
    return model

def map_color(color_id):
    colors = {1: (1, 0, 0, 1), 2: (0, 1, 0, 1), 3: (0, 0, 1, 1)}
    return colors.get(color_id, (0.5, 0.5, 0.5, 1))

def generate_new_dna(num_limbs=3):
    new_dna = []
    for _ in range(num_limbs):
        length = np.random.uniform(1.0, 3.0)
        thickness = np.random.uniform(0.1, 0.2)
        color_id = np.random.randint(1, 4)
        new_dna.append((length, thickness, color_id))
    return new_dna

def make_creature(dna_vector):
    BODY_RADIUS = 0.2
    BODY_SIZE = (BODY_RADIUS, BODY_RADIUS, BODY_RADIUS / 2)
    body_rgba = (0.8, 0.8, 0.8, 1)
    model = mjcf.RootElement()
    model.compiler.angle = 'radian'
    model.worldbody.add('geom', name='torso', type='ellipsoid', size=BODY_SIZE, rgba=body_rgba)
    for i, (length, thickness, color_id) in enumerate(dna_vector):
        rgba = map_color(color_id)
        theta = 2 * np.pi * i / len(dna_vector)
        limb_pos = BODY_RADIUS * np.array([np.cos(theta), np.sin(theta), 0])
        limb_site = model.worldbody.add('site', pos=limb_pos, euler=[0, 0, theta])
        limb = create_limb(length=length, thickness=thickness, rgba=rgba)
        limb_site.attach(limb)
    return model

step_counter = 0
dna_vector = generate_new_dna()
creature = make_creature(dna_vector)
arena = mjcf.RootElement()
arena.worldbody.add('geom', type='plane', size=[2, 2, .1])
arena.worldbody.add('light', pos=[0, 0, 3], dir=[0, 0, -1])
creature_pos = (0, 0, 0.5)
creature_site = arena.worldbody.add('site', pos=creature_pos, group=3)
creature_site.attach(creature).add('freejoint')
model = mujoco.MjModel.from_xml_string(arena.to_xml_string())
data = mujoco.MjData(model)

while True:
    mujoco.mj_step(model, data)
    step_counter += 1
    print(step_counter)
    if step_counter >= 200:
        dna_vector = generate_new_dna()
        creature = make_creature(dna_vector)
        # Reset the simulation with the new creature
        arena = mjcf.RootElement()
        arena.worldbody.add('geom', type='plane', size=[2, 2, .1])
        arena.worldbody.add('light', pos=[0, 0, 3], dir=[0, 0, -1])
        creature_pos = (0, 0, 0.5)
        creature_site = arena.worldbody.add('site', pos=creature_pos, group=3)
        creature_site.attach(creature).add('freejoint')
        model = mujoco.MjModel.from_xml_string(arena.to_xml_string())
        data = mujoco.MjData(model)
        step_counter = 0
    time.sleep(0.01)
