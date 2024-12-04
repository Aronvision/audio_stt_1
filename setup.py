from setuptools import setup, find_packages
import os
from glob import glob

package_name = 'audio_stt'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(),  # 패키지를 자동으로 찾습니다.
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),  # 패키지 인덱스 마커 파일 설치
        ('share/' + package_name, ['package.xml']),  # package.xml 설치
        # 실행 파일을 설치합니다.
        (os.path.join('lib', package_name), glob('scripts/*')),
    ],
    install_requires=[
        'setuptools',
        # 필요한 의존성 패키지들을 나열합니다.
    ],
    zip_safe=True,
    maintainer='Your Name',
    maintainer_email='your.email@example.com',
    description='Audio STT ROS2 Node',
    license='Your License',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'audio_stt_node = audio_stt.audio_stt_node:main',
        ],
    },
)

