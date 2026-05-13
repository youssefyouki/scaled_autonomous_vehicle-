from setuptools import find_packages, setup
import os
from glob import glob


package_name = 'vector_pkg_slam'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'),
            glob('launch/*.py')),
        (os.path.join('share', package_name, 'config'),
            glob('config/*.yaml')),
        (os.path.join('share', package_name, 'rviz'),
            glob('rviz/*.rviz')),
        (os.path.join('share', package_name, 'vector_pkg_slam/Helping_fn'),
            glob('vector_pkg_slam/Helping_fn/*.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='cm',
    maintainer_email='cm@todo.todo',
    description='TODO: Package description',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
                #'fake_odom = vector_pkg_slam.fake_odom:main',
                'Vehicle_Comm = vector_pkg_slam.Vehicle_Comm:main',
                'drive_control_keyboard = vector_pkg_slam.drive_control_keyboard:main',
                'drive_control_logitech = vector_pkg_slam.drive_control_logitech:main',
                'vehicle_kin_odom = vector_pkg_slam.vehicle_kin_odom:main',
        ],
    },
)
