from setuptools import find_packages, setup

if __name__ == "__main__":
    setup(
        name="psp",
        version="0.0.1",
        packages=find_packages("src"),
        package_dir={"": "src"},
    )
