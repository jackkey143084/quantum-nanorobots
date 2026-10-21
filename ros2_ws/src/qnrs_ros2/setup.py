from setuptools import setup

package_name = "qnrs_ros2"

setup(
    name=package_name,
    version="0.2.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools", "qnrs"],
    zip_safe=True,
    maintainer="QNRS maintainers",
    maintainer_email="maintainer@example.com",
    description="ROS 2 bridge to the QNRS quantum-instrument control plane.",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "quantum_bridge_node = qnrs_ros2.quantum_bridge_node:main",
        ],
    },
)
