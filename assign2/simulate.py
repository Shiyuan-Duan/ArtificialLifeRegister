import dm_control.mujoco
import mujoco.viewer
import numpy as np
import time

model = dm_control.mujoco.MjModel.from_xml_path("body.xml")
data = dm_control.mujoco.MjData(model)

steps_num = 6000
steps_time = 0.01
freq_leg_movement = 2

with mujoco.viewer.launch_passive(model, data) as viewer:
    viewer.cam.azimuth = 90
    viewer.cam.elevation = -10
    viewer.cam.distance = 5.0
    viewer.cam.lookat[:] = [0.0, 0.0, 0.75]


    for s in range(steps_num):
        angle = np.sin(2 * np.pi * freq_leg_movement * s * steps_time)

        for i in range(4):
            data.ctrl[i] = angle * (-5 if i % 2 == 0 else 5)

        dm_control.mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(steps_time)
