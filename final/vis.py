from dm_control import mujoco, mjcf
import numpy as np
import time
import mujoco.viewer
import pickle as pk
def generate_random_dna(num_limbs=3):
    dna_vector = []
    for _ in range(num_limbs):
        length = np.random.uniform(0.1, 0.5) 
        thickness = np.random.uniform(0.01, 0.05) 
        color_id = np.random.randint(1, 4)  
        dna_vector.append((length, thickness, color_id))
    return dna_vector

def create_limb(length, thickness, rgba):
    model = mjcf.RootElement()
    model.default.joint.damping = 2
    model.default.joint.type = 'hinge'
    model.default.geom.type = 'capsule'
    model.default.geom.rgba = rgba

    # Limb:
    limb = model.worldbody.add('body')
    limb_joint = limb.add('joint', axis=[0, 1, 1])
    limb.add('geom', fromto=[0, 0, 0, length, 0, 0], size=[thickness, length])

    # Position actuators:
    model.actuator.add('position', joint=limb_joint, kp=10)
    
    return model

def map_color(color_id):
    colors = {
        3: (1, 0, 0, 1),  # Red
        2: (0, 1, 0, 1),  # Green
        1: (0, 0, 1, 1)   # Blue
    }
    return colors.get(color_id, (0.5, 0.5, 0.5, 1))

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


# dna_vector = [(0.2, 0.01, 1), (1.5, 0.02, 2), (0.25, 0.05, 3)]
EPOCH = 16
with open('epoch_results.pkl', 'rb') as f:
    results = pk.load(f)

r = results[EPOCH]
num_limbs = 4
dna_vector = r['winner_dna']
creature = make_creature(dna_vector)


arena = mjcf.RootElement()
arena.worldbody.add('geom', type='plane', size=[50, 50, .1])
arena.worldbody.add('light', pos=[0, 0, 3], dir=[0, 0, -1])

creature_pos = (0, 0, 0.5)
creature_site = arena.worldbody.add('site', pos=creature_pos, group=3)
creature_site.attach(creature).add('freejoint')


model = mujoco.MjModel.from_xml_string(arena.to_xml_string())
data = mujoco.MjData(model)
viewer_handle = mujoco.viewer.launch_passive(model, data)

# Run the simulation
# Assuming actuators are named or indexed in a way you can reference them
t = 0
while True:
    t += 0.01 

    wiggle_amount = np.sin(t)  
    sine_component = np.sin(t)
    cosine_component = np.cos(t)
    for i, dna in enumerate(dna_vector):
        # data.ctrl[i] = wiggle_amount * dna[-1] * 10
        actuator_index_sine = i * 2
        actuator_index_cosine = i * 2 + 1

        # Set the actuator controls for the circular motion
        # You would need to adjust the amplitude (e.g., dna[-1] * 10) based on your simulation specifics
        data.ctrl[i] = sine_component * dna[-1] * 10
        data.ctrl[i] = cosine_component * dna[-1] * 10
    
    mujoco.mj_step(model, data)
    viewer_handle.sync()
    time.sleep(0.01)

