import numpy as np
import pandas as pd
import shapefile as shp
import matplotlib.pyplot as plt
import seaborn as sns
from math import radians, cos, sin, asin, sqrt
from tqdm import tqdm

sns.set(style='whitegrid', palette='pastel', color_codes=True)
sns.mpl.rc('figure', figsize=(10, 6))

shp_path = "./Municipio.shp"
sf = shp.Reader(shp_path)


def read_shapefile(sf):
    """
    Read a shapefile into a Pandas dataframe with a 'coords' 
    column holding the geometry information. This uses the pyshp
    package
    """
    fields = [x[0] for x in sf.fields][1:]
    records = sf.records()
    shps = [s.points for s in sf.shapes()]
    df = pd.DataFrame(columns=fields, data=records)
    df = df.assign(coords=shps)
    df = df.assign(owner=df.nome)
    df = df.assign(centre=[get_city_centre(df, city_name)
                           for city_name in df['nome']])
    return df


def get_city_coordinates(df, city_name):
    """
    From a city_name, return its XY coordinates
    """
    # Fetch the shape coordinates
    shape = df[df.nome == city_name].coords.to_numpy()[0]
    x_lon = [point[0] for point in shape]
    y_lat = [point[1] for point in shape]

    return x_lon, y_lat


def get_city_centre(df, city_name):
    """
    From a city_name, return its center coordinates
    """

    # Fetch the shape information
    return np.mean(get_city_coordinates(df, city_name), 1)


def haversine_distance(x0, y0, x1, y1):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """

    # Convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [x0, y0, x1, y1])

    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    r = 6371  # Radius of earth in kilometers

    return c * r


def distance_to_other_cities(df, city):
    """
    Returns the Haversine distance for each city
    in relation to a given city, sorted by distance
    """
    city_centre = df[df.nome == city].centre.to_numpy()[0]
    distance = [(haversine_distance(*city_centre, *df[df.nome ==
                                                      other_city].centre.to_numpy()[0]), other_city,) for other_city in df.nome]

    return sorted(distance, key=lambda x: x[0])


# War code
print("Reading and parsing shapefile")
df = read_shapefile(sf)

# Precompute the distances
DISTANCES = {}

for city_name in df['nome']:
    print("Pre-Computing distance to {}".format(city_name))
    DISTANCES[city_name] = distance_to_other_cities(df, city_name)

winners = {}
time = []
for i in tqdm(range(10000)):
    # Re-fetch the dataset
    df = df.assign(owner=df.nome)

    # Run the simulation
    runners = []
    while len(df['owner'].unique()) > 1:

        # Update quantity of runners
        runners.append(len(df['owner'].unique()))

        # Get the reference to the attacking city
        attack_city = df.sample()

        # Get the reference to the attacking city owner
        attack_city_owner = df[df.nome == attack_city.owner.to_numpy()[0]]

        # Get the attacking city name
        attack_name = attack_city.nome.to_numpy()[0]

        # Get the distance to the other cities
        distance_to_cities = DISTANCES[attack_name]

        # Get the names from all cities under the same owner
        owned_by_same = df[df.owner == attack_city_owner.nome.to_numpy()[
            0]].nome.to_numpy()

        # Get the name of the first city, in distance, which is not owned by the same guy
        defend_name = next(
            city[1] for city in distance_to_cities if city[1] not in owned_by_same and city[1])

        # Get its correspondent reference
        defend_city = df[df.nome == defend_name]

        # Get its owner
        defend_city_owner = df[df.nome == defend_city.owner.to_numpy()[0]]

        # Print information
        # print("{} conquers {} from {} through {}".format(attack_city_owner.nome.to_numpy()[0], defend_name, defend_city_owner.nome.to_numpy()[0], attack_name))

        # Update the city owner
        df.loc[df['nome'] == defend_name, ['owner']] = attack_city_owner.nome.to_numpy()[
            0]

    winner_name = df['owner'].to_numpy()[0]
    winners[winner_name] = winners[winner_name] + \
        1 if winner_name in winners else 1
    time.append(len(runners))


# Average of time to run
print("Average of {:.2f} steps, this is {:.2f} days".format(
    np.mean(time), np.mean(time) / 24))

# Plot winners
plt.bar(list(winners.keys()), list(winners.values()))
