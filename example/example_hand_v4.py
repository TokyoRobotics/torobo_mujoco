import mujoco
import mujoco.viewer
import time

m = mujoco.MjModel.from_xml_path("../robots/hand_v4/hand_v4.xml")
d = mujoco.MjData(m)

with mujoco.viewer.launch_passive(m, d) as viewer:
    start = time.time()

    while viewer.is_running():
        step_start = time.time()
        mujoco.mj_step(m, d)
        viewer.sync()

        time_until_next_step = m.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)
