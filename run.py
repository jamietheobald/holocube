#! /usr/bin/env python
# import pyglet
# import holocore.hc as hc
#
# hc.control.start()
# hc.window.start(config_file='test_viewport.config')
# hc.arduino.start('dummy')
# hc.scheduler.start(hc.window, hc.control,
#                    randomize=False, default_rest_time=.1)
# hc.scheduler.load_dir('experiments', suffix=('exp.py', 'rest.py'))
# print('ready')
#
# pyglet.app.run()


#! /usr/bin/env python
import argparse
import pyglet
import holocore.hc as hc

def setup_camera(mode: str):
    """
    mode: 'off' | 'on' | 'required'
    Returns a camera object or None.
    """
    if mode == "off":
        print("camera: disabled")
        return None

    try:
        # Keep this import inside the function so missing SDKs do not kill stimulus-only mode.
        from holocore.camera_blackfly import BlackflyCamera
    except Exception as e:
        if mode == "required":
            raise
        print(f"camera: unavailable (import failed): {e}")
        return None

    try:
        cam = BlackflyCamera()
        cam.start()
        print("camera: started")
        return cam
    except Exception as e:
        if mode == "required":
            raise
        print(f"camera: unavailable (start failed): {e}")
        return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", choices=["off", "on", "required"], default="off")
    parser.add_argument("--config", default="test_viewport.config")
    args = parser.parse_args()

    hc.control.start()
    hc.window.start(config_file=args.config)
    hc.arduino.start("dummy")

    # optional camera
    cam = setup_camera(args.camera)
    if cam is not None:
        # simplest: attach to control so it can draw preview and read analysis
        hc.control.attach_camera(cam)  # you will add this method in control.py

        # clean shutdown when windows close
        @hc.control.event
        def on_close():
            try:
                cam.stop()
            finally:
                hc.control.close()

    hc.scheduler.start(
        hc.window, hc.control,
        randomize=False, default_rest_time=0.1
    )  # scheduler already expects window and control :contentReference[oaicite:2]{index=2}

    hc.scheduler.load_dir("experiments", suffix=("exp.py", "rest.py"))
    print("ready")
    pyglet.app.run()

if __name__ == "__main__":
    main()