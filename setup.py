from setuptools import find_packages, setup

test_requirements = ["pytest"]
docs_requirements = [
    "Sphinx==3.2.1",
    "sphinxcontrib-websupport==1.2.4",
    "sphinx_rtd_theme",
]

setup(
    name="kyle-calibration",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.8",
    license="MIT",
    url="https://github.com/appliedAI-Initiative/kyle",
    include_package_data=True,
    version="0.1.1",
    description="appliedAI classifier calibration library",
    install_requires=open("requirements.txt").readlines(),
    setup_requires=["wheel"],
    tests_require=test_requirements,
    extras_require={"test": test_requirements, "docs": docs_requirements},
    author="appliedAI",
)
