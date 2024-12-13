from setuptools import find_packages, setup
from typing import List

def getrequirements(filepath:str)->List[str]:
    '''
    #End to End Ml Project requirements installer(Metro Interstate Prediction)
    '''
    HYPEN = "-e ."
    requirements = []
    with open(filepath) as fileobj:
        requirements = fileobj.readlines()
        requirements = [i.replace("\n", "") for i in requirements]
        if HYPEN in requirements:
            requirements.remove(HYPEN)
    return requirements

setup(
name = "Metro_Interstate_Prediction",
version = '0.0.1',
author= "Rohith Obillaneni",
author_email= 'rohithobillaneni92@gmail.com',
packages= find_packages(),
install_requires = getrequirements('requirements.txt')
)
