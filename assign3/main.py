dna_vector = [(2, 0.1, 1), (1.5, 0.2, 2), (2.5, 0.15, 3)]
colors = {
    1: (1, 0, 0, 1),  # Red
    2: (0, 1, 0, 1),  # Green
    3: (0, 0, 1, 1)   # Blue
}

from dm_control import mujoco, mjcf
import numpy as np
import time
import mujoco.viewer


def create_limb(length, thickness, rgba):
    model = mjcf.RootElement()
    model.default.joint.damping = 2
    model.default.joint.type = 'hinge'
    model.default.geom.type = 'capsule'
    model.default.geom.rgba = rgba

    # Limb:
    limb = model.worldbody.add('body')
    limb_joint = limb.add('joint', axis=[0, 0, 1])
    limb.add('geom', fromto=[0, 0, 0, length, 0, 0], size=[thickness, length])

    # Position actuators:
    model.actuator.add('position', joint=limb_joint, kp=10)
    
    return model

def map_color(color_id):
    colors = {
        1: (1, 0, 0, 1),  # Red
        2: (0, 1, 0, 1),  # Green
        3: (0, 0, 1, 1)   # Blue
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


dna_vector = [(0.2, 0.01, 1), (0.15, 0.02, 2), (0.25, 0.05, 3)]
creature = make_creature(dna_vector)


arena = mjcf.RootElement()
arena.worldbody.add('geom', type='plane', size=[2, 2, .1])
arena.worldbody.add('light', pos=[0, 0, 3], dir=[0, 0, -1])

creature_pos = (0, 0, 0.5)
creature_site = arena.worldbody.add('site', pos=creature_pos, group=3)
creature_site.attach(creature).add('freejoint')


model = mujoco.MjModel.from_xml_string(arena.to_xml_string())
data = mujoco.MjData(model)
viewer_handle = mujoco.viewer.launch_passive(model, data)

# Run the simulation
while True:
    mujoco.mj_step(model, data)
    viewer_handle.sync()
    time.sleep(0.01)
