#! /usr/bin/env python
import argparse

import pyglet

import holocore.hc as hc
from holocore import launch


def main():
    parser = argparse.ArgumentParser()
    launch.add_arguments(parser)
    args = parser.parse_args()

    frame_rate = 120

    launch.start_holocube(
        hc,
        config_file=args.config,
        camera=args.camera,
        frame_rate=frame_rate,
    )
    print("ready")

    pyglet.app.run(interval=None)


if __name__ == "__main__":
    main()
