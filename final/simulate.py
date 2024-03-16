

from dm_control import mujoco, mjcf
import numpy as np
import time
import mujoco.viewer
import pickle
import pickle
def generate_random_dna(num_limbs=4):
    dna_vector = []
    for _ in range(num_limbs):
        length = np.random.uniform(0.01, 0.5) 
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




def simulate(dna_vector, data, model, viewer_handle):
    max_height = -1*np.inf
    vis = (viewer_handle is not None)
    for t in range(3000):
        t/=100
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
        creature_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, 'creature_site')
        creature_position = data.site_xpos[creature_site_id]

        height = np.linalg.norm(np.array(creature_position))
        max_height = max(max_height, height)
        
        mujoco.mj_step(model, data)
        if vis:
            viewer_handle.sync()
            time.sleep(0.01)
    return max_height


# def mutate_dna(dna_vector):

#     limb_index = np.random.randint(3)

#     limb = dna_vector[limb_index]

#     limb_list = list(limb)
#     mutation_index = np.random.randint(3)  # Selects which attribute to mutate (0, 1, or 2)
#     if mutation_index == 0:
#         limb_list[0] = np.random.uniform(0.1, 1.0)  # Mutate length
#     elif mutation_index == 1:
#         limb_list[1] = np.random.uniform(0.01, 0.01)  # Mutate thickness
#     else:
#         limb_list[2] = np.random.randint(1, 4)  # Mutate color_id

    
#     dna_vector[limb_index] = tuple(limb_list)

#     return dna_vector

def mutate_dna(dna_vector):
    mutated_dna = []
    for limb in dna_vector:
        limb_list = list(limb)
        mutation_index = np.random.randint(3)  
        if mutation_index == 0:
            limb_list[0] = np.random.uniform(0.01, 0.5) 
        elif mutation_index == 1:
            limb_list[1] = np.random.uniform(0.01, 0.05) 
            limb_list[2] = np.random.randint(1, 4) 
        mutated_dna.append(tuple(limb_list))
    return mutated_dna


START_DNA = [(0.01, 0.01, 1), (0.01, 0.01, 1), (0.01, 0.01, 1), (0.01, 0.01, 1)]


def main():
    epoch_results = []

    for epoch in range(20):
        global_max_height = -1 * np.inf
        winner_dna = START_DNA
        epoch_winner = START_DNA
        all_mobility = []

        for i in range(100):
            # Recreate the arena for each new creature
            arena = mjcf.RootElement()
            arena.worldbody.add('geom', type='plane', size=[50, 50, .1])
            arena.worldbody.add('light', pos=[0, 0, 3], dir=[0, 0, -1])

            # Generate or mutate DNA

            if epoch == 0:
                dna_vector = START_DNA
            else:
                dna_vector = mutate_dna(epoch_winner)
            # if epoch_winner is None:
            #     dna_vector = START_DNA
            # else:
            #     dna_vector = mutate_dna(epoch_winner)
            
            # Create creature and model
            creature = make_creature(dna_vector)
            creature_site = arena.worldbody.add('site', pos=(0, 0, 0.5), group=3)
            creature_site.attach(creature).add('freejoint')
            model = mujoco.MjModel.from_xml_string(arena.to_xml_string())
            data = mujoco.MjData(model)

            # Assume simulate is a function that returns the height (mobility) of the creature
            curr_height = simulate(dna_vector=dna_vector, data=data, model=model, viewer_handle=None)
            all_mobility.append(curr_height)
            
            # Determine the winner
            if curr_height > global_max_height:
                winner_dna = dna_vector
                global_max_height = curr_height
                # print(f'assigning epoch winner with :{dna_vector}')

            epoch_winner = winner_dna

        average_mobility = np.mean(all_mobility)
        print(f"Epoch {epoch}: Max Height = {global_max_height} | Average Mobility: {average_mobility}")

        # Store the results of the epoch
        epoch_results.append({
            'epoch': epoch,
            'max_height': global_max_height,
            'winner_dna':epoch_winner,
            'average_mobility': average_mobility
        })

    # Save the results to a pickle file
    with open('epoch_results.pkl', 'wb') as file:
        pickle.dump(epoch_results, file)

main()
