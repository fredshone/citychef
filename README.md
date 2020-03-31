# City Chef

*Welcome to City Chef, here you can build your own city.*

City Chef is a collection of methodologies for essentially just randomly generating data, 
but where the data generating process (DGP) is like (within reason) a real realistic city. The 
code is intended for use in an interactive (ipython) environment. Data can
 be generated for all sorts of city features:

**Facilities** (activity locations, such as households, workplaces and hospitals)

**Road Networks**

**Road Transit Routes**

**Rail Transit Network and Routes**

**Zones** statistical zones based on the city density

**Population** (agents, with consistent household attributes)

**Activity Plans** (simple activity based plans for each agent)

### Motivation

The motivation behind this project is to fill a gap in availability of city data - because 
of privacy issues, commercial constraints, or because the quality and quantity of data required 
is not available. We also use City Chef for making small data for tests or toy examples, 
where we'd rather not use full city scale data. Included in this project are two such example 
applications:

**1. [Census & Travel Survey Generator](https://github.com/fredshone/citychef/blob/master/census-and-travel-survey-generation.ipynb)**

Build a full city with facilities, statistical areas, road and transit networks. Use this city to 
generate households of persons with complex underlying distributions of attributes. Output 
household travel survey data with simple activity and mode choice. Output census marginal 
statistics. Output commuter OD matrix.

We use this generator for population synthesis experiments.

![Example](https://github.com/fredshone/citychef/blob/master/images/city.png)

**2. [Census & Travel Survey Generator](https://github.com/fredshone/citychef/blob/master/osm-and-gtfs-generation.ipynb)**

Build a city with a road network and generate bus routes. Apply spatial noise to the networks. 
Output the road network to OSM format. Output the bus routes to GTFS.

We use this generator for making test case data for our big network combining tools.

![Example](https://github.com/fredshone/citychef/blob/master/images/test.png)

### Installation

python3.7
ipython7.0.1


after git clone:
```
cd citychef
pip3 install -r requirements.txt
pip3 install -e .
```

*This project is WIP and more a collection of methods than a API, so we apologise in advance for 
environment pains.*

`pyproj`: The GeoPandas library requires pyproj as a dependency. This can be a bit of a pain to 
install. For Mac OSX, activate the environment City Chef lives in and run the following commands 
before installation:

```
pip3 install cython
pip3 install git+https://github.com/jswhit/pyproj.git
```

On Windows, pre-compiled wheels of pyproj can be found on this page. Manually install the correct
 pyproj wheel within your environment using pip.

### Critique/ideas/TODO

#### Technical

- no doc strings
- very little logging
- no tests
- needs epsg conversions (osm and gtfs)
- needs better structure/API
- many of the classes and methods share features
- bad module names
- speed vs detail trade-off

#### Theoretical

- accessibility calculation is simplistic
- activity plan synthesis is only simple tour based
- facility locations only contain a single facility
- transit route synthesis is pretty bad, especially for trains

More broadly City Chef uses directed tree like causation for the DGPs. This is primarily for 
simplicity/speed. But means that circular causation is only synthesised one way, for 
example, the feedback between facility locations and network.
