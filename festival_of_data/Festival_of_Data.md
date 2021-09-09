# Festival of Data

Welcome to our two new cities; `city_A` and `city_B`. Designed by two different planners, in some ways they are remarkably similar but in others very different.

Both cities have around 2000 households, road and transit networks, but their forms, functions and people have changed over time, so that today they are sometimes quite different.

## The Data

For each city you will find data describing the form of the cities, their functioning and the people living there. Data is in various forms as described below:

* `bus_networks` - (spatial)
* `car_networks` - (spatial)
* `rail_networks` - (spatial)
* `bus_gtfs` - (spatio-temporal) [General Transit Feed Specification](https://developers.google.com/transit/gtfs), a description of bus stops and vehicle movements
* `rail_gtfs` - (spatio-temporal) [General Transit Feed Specification](https://developers.google.com/transit/gtfs), a description of rail stops and vehicle movements
* `building_locations` - (spatial) Locations for every home, workplace, shop, school and hospital
* `population_survey` - (tabular) Person and Household attributes of the entire population, including how they travel
* `administrative_areas` - (spatial) Large administrative geometries
* `administrative_zones` - (spatial) Detailed administrative geometries
* `zone_commuter_freq`- (tabular) Summary of people movements for work and education activities

All spatial data is presented as lat/lon eg: WGS:84.

The datasets for each city generally linkable, for example using their locations or their ids.

## Is this data safe?

Yes. This data is all synthetic, built with a tool we call CityChef. CityChef is crafted from a lot of interconnected algorithms, designed to replicate data distributions and crucially relationships in real cities.

## More about City Chef

We've been researching and building new approaches to modelling human behaviour in cities and countries. With the aim of helping design better and farer policies and infrastructure for the future. A hugely important consideration of our work is the complexity and hetrogeneity of humans. No one is the same and no one should be treated the same when trying to plan the future.

But this often puts us in an awkward possition for our research - data about people and their behaiour is useful to us, but accessing it is hard, the quality is often dubious and we prefer not to work with sensitive personal data.

City Chef is our tool for working around this data challenge. City Chef builds fake data for use by researchers and modellers looking to develop and test new approaches. City Chef has some key aims:

1. Output data that is often not obsevred or measured
2. Output data in useful formats
3. Use representative physical components, such as networks
4. Use representative probabilistic components, such as age distributions

and where (4) fails: (5) Use representative complexity in the distributions

You can read more about CityChef [here](https://medium.com/arupcitymodelling/def-city-chef-a72326cceddb).
