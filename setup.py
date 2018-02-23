from setuptools import setup


setup(
    name="conceptnet-sme",
    version='0.1',
    maintainer='Luminoso Technologies, Inc.',
    maintainer_email='rspeer@luminoso.com',
    license="Apache License 2.0",
    url='http://github.com/LuminosoInsight/conceptnet-sme',
    platforms=["any"],
    description="Models relations between terms in general knowledge",
    packages=['conceptnet_sme'],
    install_requires=['numpy', 'conceptnet'],
    # also requires pytorch but pytorch doesn't install with pip
    zip_safe=False,
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
)
