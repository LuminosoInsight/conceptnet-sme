from setuptools import setup


setup(
    name="general-relativity",
    version='0.1',
    maintainer='Luminoso Technologies, Inc.',
    maintainer_email='rspeer@luminoso.com',
    license="Proprietary",
    url='http://github.com/LuminosoInsight/general-relativity',
    platforms=["any"],
    description="Models relations between terms in general knowledge",
    packages=['general_relativity'],
    install_requires=['numpy', 'conceptnet'],
    # also requires pytorch but pytorch doesn't install with pip
    zip_safe=False,
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
)
