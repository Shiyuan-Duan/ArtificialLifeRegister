import dm_control.mujoco
import mujoco.viewer
import time
import numpy as np


model = dm_control.mujoco.MjModel.from_xml_path("snowman.xml")
data = dm_control.mujoco.MjData(model)


with mujoco.viewer.launch_passive(model, data) as viewer:
    radius = 1.0
    speed = 0.1
    for i in range(10000):
        if viewer.is_running():

            x = radius * np.cos(i * speed)
            y = radius * np.sin(i * speed)


            data.qpos[0] = x
            data.qpos[1] = y

            dm_control.mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(0.01)
        else:
            break

    viewer.close()
