import numpy as np
import pandas as pd
import shapefile as shp
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import seaborn as sns
from tqdm import tqdm
from math import radians, cos, sin, asin, sqrt
import os
import json
import gc
from pprint import pprint as pp
import glob

sns.set(style='white', palette='pastel', color_codes=True)
sns.mpl.rc('figure', figsize=(10, 6))

def read_shapefile(sf):
    """
    Read a shapefile into a Pandas dataframe with a 'coords' 
    column holding the geometry information. This uses the pyshp
    package
    """

    print("Generating Dataframe")
    if os.path.isfile('./dataframe.h5'):
        print("Reading Dataframe from HDF5")
        df = pd.read_hdf('./dataframe.h5')
    else:
        df = pd.DataFrame(columns=[x[0] for x in sf.fields][1:], data=sf.records())
        df = df.assign(coords=[s.points for s in sf.shapes()])
        df = df.assign(owner=df.nome)
        df = df.assign(centre=[get_city_centre(df, city_name)
                            for city_name in df['nome']])
        df = df.assign(color=[np.random.rand(3,) / 2 +
                            0.5 for city_name in df['nome']])
        df = df.assign(protected=[int(np.random.rand() * 10) for city_name in df['nome']])

        # Save to H5
        df.to_hdf('./dataframe.h5', 'df')

    # If already run, update the fields
    if os.path.isfile('./dataframe.json'):
        print("Reading owner and protected from JSON file")
        df.update(pd.read_json('./dataframe.json', orient='records'))
    
    return df


def get_city_coordinates(df, city_name):
    """
    From a city_name, return its XY coordinates
    """
    # Fetch the shape coordinates
    coords = df[df.nome == city_name].coords.to_numpy()[0]
    x_lon = [point[0] for point in coords]
    y_lat = [point[1] for point in coords]

    return x_lon, y_lat

def get_city_centre(df, city_name):
    """
    From a city_name, return its center coordinates
    """

    # Fetch city coordinates
    coord_x, coord_y = get_city_coordinates(df, city_name)

    # Compute signed area
    A = 0
    for i in range(len(coord_x)):
        A += coord_x[i] * coord_y[(i + 1) % len(coord_y)] - coord_x[(i + 1) % len(coord_x)] * coord_y[i]
    A /= 2

    # Compute C_x
    C_x = 0
    C_y = 0
    for i in range(len(coord_x)):
        C_x += (coord_x[i] + coord_x[(i + 1) % len(coord_x)]) * (coord_x[i] * coord_y[(i + 1) % len(coord_y)] - coord_x[(i + 1) % len(coord_x)] * coord_y[i])
        C_y += (coord_y[i] + coord_y[(i + 1) % len(coord_x)]) * (coord_x[i] * coord_y[(i + 1) % len(coord_y)] - coord_x[(i + 1) % len(coord_x)] * coord_y[i])
    C_x /= 6 * A
    C_y /= 6 * A

    # Fetch the shape information
    return C_x, C_y


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


def plot_city(df, city_name, color='w', print_text=False, city_linewidth=0.03, **kwargs):
    """ Plots a single shape """

    # Fetch and plot the shape with its contour
    x_lon, y_lat = get_city_coordinates(df, city_name)
    plt.fill(x_lon, y_lat, facecolor=color, zorder=-2)
    plt.plot(x_lon, y_lat, 'w', linewidth=city_linewidth)
    

    # Configure the text plotting
    if print_text:
        x_center, y_center = get_city_centre(df, city_name)
        plt.text(x_center, y_center, city_name,
                 va='center', ha='center', **kwargs)


def sizeof_fmt(num, suffix='B'):
    ''' By Fred Cirera, after https://stackoverflow.com/a/1094933/1870254'''

    for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
        if abs(num) < 1024.0:
            return "%3.1f%s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f%s%s" % (num, 'Yi', suffix)

def plot_arrow(a, b, crossed=False, **kwargs):
    '''
    Plot an arrow from city A to city B
    '''

    start = get_city_centre(df, a)
    end = get_city_centre(df, b)

    x = start[0]
    y = start[1]
    dx = end[0] - x
    dy = end[1] - y

    # Draw an 'X' in the arrow, if crossed
    if crossed:
        plt.text(x + dx/2, y + dy/2, 'X', va='center', ha='center', fontsize=8) \
           .set_path_effects([PathEffects.withStroke(linewidth=2, foreground='w')])

    return plt.arrow(x, y,
                     (dx - 0.007) if dx > 0 else (dx + 0.007),
                     (dy - 0.007) if dy > 0 else (dy + 0.007),
                     width=0.005,
                     head_length=0.02,
                     color='k',
                     **kwargs)

def plot_map(df, x_lim=None, y_lim=None, figsize=(16, 13), attack = None, city_linewidth=0.03, **kwargs):
    '''
    Plot map with lim coordinates, and the cities asked with their correspondent color
    '''

    # Configure the plot
    fig, ax = plt.subplots(figsize=figsize)

    # Plot the cities
    for owner_name in df.owner.unique():
        owner = df[df['nome'] == owner_name].iloc[0]
        owned_by = df[df['owner'] == owner_name]
        for index, row in owned_by.iterrows():
            plot_city(df, row.nome, owner.color, city_linewidth=city_linewidth, **kwargs)
    gc.collect() # Call Garbage Collector explicitly

    # Configure the map size
    if (x_lim != None) and (y_lim != None):
        plt.xlim(x_lim)
        plt.ylim(y_lim)

    # Plot texts
    owners = df.owner.unique()
    np.random.shuffle(owners) # Shuffle in place

    # Predefined first owners (always plotted, if alive)
    ALWAYS_ON_TOP = ['Porto Alegre',
                     attack['attack'] if attack is not None else '',
                     attack['defend'] if attack is not None else '']

    for city in ALWAYS_ON_TOP:   
        if city in owners:
            owners = np.concatenate(([city], owners))

    texts_bb = []
    for owner_name in owners:
        owned_by = df[df['owner'] == owner_name]
        x_center, y_center = np.mean(
            [get_city_centre(df, owned_by_name) for owned_by_name in owned_by.nome], 0)
        txt = plt.text(x_center, y_center, owner_name,
                    va='center', ha='center', **kwargs)
        txt.set_path_effects(
            [PathEffects.withStroke(linewidth=3, foreground='w')])
        bb_transformed = txt.get_window_extent(renderer = fig.canvas.get_renderer()) \
                            .transformed(ax.transData.inverted())
        
        # Tests interesction
        for bb in texts_bb:
            if bb.overlaps(bb_transformed):
                txt.remove()
                break
        else:
            texts_bb.append(bb_transformed)
    
    # Plot arrow for attack
    arrow = None
    if attack != None:
        arrow = plot_arrow(attack['attack'], attack['defend'], crossed=not attack['success'], alpha=0.8, zorder=100)


    return fig, ax

def run(*, times=1):
    '''
    Run function
    '''
    cities = []

    for i in range(times):
        if len(df.owner.unique()) > 1:
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

            # Check city protection
            if defend_city.protected.to_numpy()[0] > 0:
                # Print information
                print("{} tried to conquer {} from {} through {}, but failed".format(attack_city_owner.nome.to_numpy()[
                    0], defend_name, defend_city_owner.nome.to_numpy()[0], attack_name))

                cities.append({'attack': attack_city_owner.nome.to_numpy()[0],
                               'defend': defend_city_owner.nome.to_numpy()[0],
                               'success': False})
            else:
                # Print information
                print("{} conquers {} from {} through {}".format(attack_city_owner.nome.to_numpy()[
                    0], defend_name, defend_city_owner.nome.to_numpy()[0], attack_name))

                # Update the city owner
                df.loc[df['nome'] == defend_name, ['owner']] = attack_city_owner.nome.to_numpy()[
                    0]

                cities.append({'attack': attack_city_owner.nome.to_numpy()[0],
                               'defend': defend_city_owner.nome.to_numpy()[0],
                               'success': True})
            
            # Decrease protectedness
            df.loc[df['protected'] > 0, ['protected']] = df[df['protected'] > 0].protected - 1
    return cities


### CHART GENERATION ###

# Read the dataframe
df = read_shapefile(shp.Reader("./Municipio.shp"))

# Precompute the distances
DISTANCES = {}
if os.path.isfile('./distances.json'):
    print("Fetching distances from cached file")
    with open('./distances.json', 'r') as f:
        DISTANCES = json.load(f)
else:
    print("Computing distances")
    with tqdm(df['nome']) as t:
        for city_name in t:
            t.set_description(city_name)
            DISTANCES[city_name] = distance_to_other_cities(df, city_name)

    # Write the distances to a file
    print("Writing distances to cache file")
    with open('distances.json', 'w') as f:
        json.dump(DISTANCES, f)

# Get the current iteration number
basepath = './figures'
counter = max([0] + [int(f.split('\\')[-1].split('.')[0].split('_')[-1]) for f in glob.glob(basepath + "/*.jpg")])

# Run the code
while len(df.owner.unique()) > 1:
    counter += 1

    # Run the attacks
    attacks = run()

    # Plot the map
    fig, ax = plot_map(df, figsize=(15, 12), attack=attacks[-1], fontsize=9)

    # Call Garbage Collector explicitly
    gc.collect()

    # Remove axis, ticks and remove padding
    sns.despine(top=True, right=True, left=True, bottom=True)
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
    ax.set_yticks([])
    ax.set_xticks([])

    # Save figure
    plt.savefig("{}/{}.jpg".format(basepath, counter), dpi=200)
    print("Saved figure to {}/{}.jpg".format(basepath, counter))
    plt.close()

    # Plot the zoomed map
    x_lim, y_lim = (float("inf"), float("-inf")), (float("inf"), float("-inf"))
    for city_name in df[(df.owner == attacks[-1]['attack']) | (df.owner == attacks[-1]['defend'])].nome:
        coord_x, coord_y = get_city_coordinates(df, city_name)
        x_lim = (min([x_lim[0]] + coord_x), max([x_lim[1]] + coord_x))
        y_lim = (min([y_lim[0]] + coord_y), max([y_lim[1]] + coord_y))
    x_lim = (x_lim[0] - 0.02, x_lim[1] + 0.02)
    y_lim = (y_lim[0] - 0.02, y_lim[1] + 0.02)
    fig, ax = plot_map(df,
                       figsize=(15, 12),
                       attack=attacks[-1],
                       fontsize=30,
                       city_linewidth=4.,
                       x_lim=x_lim,
                       y_lim=y_lim)

    # Call Garbage Collector explicitly
    gc.collect()

    # Remove axis, ticks and remove padding
    sns.despine(top=True, right=True, left=True, bottom=True)
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
    ax.set_yticks([])
    ax.set_xticks([])

    # Save figure
    plt.savefig("{}/zoom_{}.jpg".format(basepath, counter), dpi=200)
    print("Saved figure to {}/zoom_{}.jpg".format(basepath, counter))
    plt.close()

    # Call Garbage Collector explicitly
    gc.collect()
    
    # Update Dataframe HDF5 file
    print("Saving Updatable Dataframe Information to JSON")
    df[['nome', 'owner', 'protected']].to_json('./dataframe.json', orient='records') # Save to HDF5
    
    # Call Garbage Collector explicitly
    gc.collect()
