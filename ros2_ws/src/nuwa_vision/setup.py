from setuptools import setup
from glob import glob
import os

package_name = 'nuwa_vision'
setup(name=package_name, version='0.1.0', packages=[package_name],
      data_files=[('share/ament_index/resource_index/packages', ['resource/' + package_name]),
                  ('share/' + package_name, ['package.xml']),
                  (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
                  (os.path.join('share', package_name, 'launch'), glob('launch/*.py'))],
      install_requires=['setuptools'], zip_safe=True,
      entry_points={'console_scripts': ['rgbd_detector = nuwa_vision.rgbd_detector_node:main']})
