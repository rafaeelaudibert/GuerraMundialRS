# Core imports
import gc
import glob
import json
import os
from math import asin, cos, radians, sin, sqrt
from pprint import pprint as pp

# Library imports
import geopandas
import matplotlib.patheffects as PathEffects
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from descartes.patch import PolygonPatch
from matplotlib.collections import PatchCollection
from tqdm import tqdm

# Configure SNS
sns.set(style='white', palette='pastel', color_codes=True)
sns.mpl.rc('figure', figsize=(10, 6))


def read_shapefile():
    """
    Read a shapefile into a Geopandas dataframe
    """

    print("Generating Dataframe")
    df = geopandas.read_file('./Municipio.shp')

    if os.path.isfile('./guerra.json'):  # If already ran, update the fields
        print("Reading owner, color and protected from JSON file")
        json_df = pd.read_json('./guerra.json', orient='records')
        df[['owner', 'color', 'protected', 'ranking']] = json_df[[
            'owner', 'color', 'protected', 'ranking']]
    else:  # Create new fields
        print("Configuring owner, color and protected")
        df['owner'] = df.nome
        df['color'] = [np.random.rand(3,) / 2 + 0.5 for city_name in df.nome]
        df['protected'] = [0 for city_name in df['nome']]
        df['ranking'] = float('nan')

    return df


def haversine_distance(p0, p1):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    """

    # Convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [p0.x, p0.y, p1.x, p1.y])

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
    city_centre = df[df.nome == city].geometry.centroid
    distance = [(haversine_distance(city_centre, df[df.nome ==
                                                    other_city].geometry.centroid), other_city,) for other_city in df.nome]

    return sorted(distance, key=lambda x: x[0])


def plot_arrow(a, b, crossed=False, **kwargs):
    '''
    Plot an arrow from city A to city B
    '''

    start = df[df.nome == a].geometry.centroid.to_numpy()[0]
    end = df[df.nome == b].geometry.centroid.to_numpy()[0]

    x, y = start.x, start.y
    dx, dy = end.x - x, end.y - y

    # Draw an 'X' in the arrow, if crossed
    if crossed:
        plt.text(x + dx/2, y + dy/2 - 0.002, 'X', va='center', ha='center', zorder=120, fontsize=kwargs.get('arrow_fontsize', 7)) \
           .set_path_effects([PathEffects.withStroke(linewidth=2, foreground='w')])

    return plt.arrow(x, y, dx, dy, width=0.005, head_length=0.02, zorder=119, color='k')
                     # (dx - 0.007) if dx > 0 else (dx + 0.007)
                     # (dy - 0.007) if dy > 0 else (dy + 0.007)


def plot_map(df, x_lim=None, y_lim=None, figsize=(16, 13), attack=None, city_linewidth=0.03, zoom=False, **kwargs):
    '''
    Plot map with lim coordinates, and the cities asked with their correspondent color
    '''

    # Configure the plot
    fig, ax = plt.subplots(figsize=figsize)

    # Plot the cities (using custom code, as the fucking geopandas doesn't work correctly)
    ax.set_aspect('equal')
    collection = PatchCollection([PolygonPatch(poly['geometry'], facecolor=poly['color'], hatch='//') for (i, poly) in df.iterrows()], match_original=True)
    ax.add_collection(collection, autolim=True)
    ax.autoscale_view()

    # Configure the map size
    if (x_lim != None) and (y_lim != None):
        plt.xlim(x_lim)
        plt.ylim(y_lim)

    # Plot texts
    owners = df.owner.unique()
    np.random.shuffle(owners)  # Shuffle in place

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
        center = df[df.owner == owner_name].geometry.unary_union.centroid
        txt = plt.text(center.x, center.y, owner_name,
                       va='center', ha='center', zorder=130 if x_lim is not None else 110, fontsize=kwargs.get('fontsize', 6))
        txt.set_path_effects(
            [PathEffects.withStroke(linewidth=3, foreground='w')])
        bb_transformed = txt.get_window_extent(renderer=fig.canvas.get_renderer()) \
                            .transformed(ax.transData.inverted())

        # Check if it is outside of the map
        outside = False
        if (x_lim != None) and (y_lim != None):
            if bb_transformed.xmin < x_lim[0] or \
                bb_transformed.xmax > x_lim[1] or \
                bb_transformed.ymin < y_lim[0] or \
                bb_transformed.ymax > y_lim[1]:
                    txt.remove()
                    outside = True

        # Tests interesction if not already outside
        if not outside:
            for bb in texts_bb:
                if bb.overlaps(bb_transformed):
                    txt.remove()
                    break
            else:
                texts_bb.append(bb_transformed)

    # Plot arrow for attack
    arrow = None
    if attack != None:
        arrow = plot_arrow(attack['attack'], attack['defend_itself'],
                           crossed=not attack['success'], alpha=0.8, zorder=100, **kwargs)

        # Add borders to attack city, and defend city
        collection = PatchCollection([PolygonPatch(df[df.nome == attack['defend_itself']].geometry.to_numpy()[0], fill=None, zorder=200, edgecolor='blue', linewidth=10 if zoom else 2)], match_original=True)
        ax.add_collection(collection, autolim=True)

        collection = PatchCollection([PolygonPatch(df[df.owner == attack['attack']].geometry.unary_union, fill=None, zorder=200, edgecolor='red', linewidth=5 if zoom else 1.5)], match_original=True)
        ax.add_collection(collection, autolim=True)
        
        if len(df[df.owner == attack['defend']]) > 0:
            collection = PatchCollection([PolygonPatch(df[df.owner == attack['defend']].geometry.unary_union, fill=None, zorder=200, edgecolor='green', linewidth=3 if zoom else 1)], match_original=True)
            ax.add_collection(collection, autolim=True)

        
        
        ax.autoscale_view()

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
                               'attack_itself': attack_name,
                               'defend_itself': defend_name,
                               'success': False})
            else:
                # Print information
                print("{} conquers {} from {} through {}".format(attack_city_owner.nome.to_numpy()[
                    0], defend_name, defend_city_owner.nome.to_numpy()[0], attack_name))

                # Update the city owner and color
                df.loc[df['nome'] == defend_name, ['owner']] = \
                    attack_city_owner.nome.to_numpy()[0]
                df.loc[df['nome'] == defend_name, 'color'] = \
                    attack_city_owner.color.to_numpy()

                cities.append({'attack': attack_city_owner.nome.to_numpy()[0],
                               'defend': defend_city_owner.nome.to_numpy()[0],
                               'attack_itself': attack_name,
                               'defend_itself': defend_name,
                               'success': True})

                # Verify if need to update ranking
                if defend_city_owner.nome.to_numpy()[0] not in list(df.owner):
                    print('Adding city to ranking')
                    df.loc[df.nome == defend_city_owner.nome.to_numpy()[0], ['ranking']] = len(df.owner.unique()) + 1

            # Decrease protectedness
            df.loc[df['protected'] > 0, ['protected']
                   ] = df[df['protected'] > 0].protected - 1
    return cities

def save_text(df, basepath, counter, attack):
    '''
        Save the tet for this attack to file
    '''
    LOCATIONS = np.array(['a querência de', 'o território de', 'as terras de', 'todos os bois de'])
    LOCATIONS_FULL = np.array(['A querência de', 'Os soldados de', 'Os peões de', 'A gurizada de'])
    CONQUER_VERB = np.array(['ataca', 'derruba', 'passa por cima de'])
    ELIMINATIONS = np.array(['está fora do jogo', 'foi brutalmente eliminado', 'está fora de combate', 'perdeu todos seus territórios', 'foi eliminado', 'tá fora da peleia'])

    # Post text
    with open("{}/{}/{}.txt".format(basepath, counter, 'post'), 'w') as f:
        first_line = f"{np.random.choice(LOCATIONS_FULL) if np.random.randint(100) >= 60 else ''} {attack['attack']} {np.random.choice(CONQUER_VERB) if np.random.randint(100) >= 75 else 'conquista'} {attack['defend_itself']}"
        
        if attack['defend_itself'] != attack['defend']: # Different defender territory and defender owner
            first_line += f" {np.random.choice(LOCATIONS) if np.random.randint(100) >= 70 else 'de'} {attack['defend']}"
        
        if attack['attack_itself'] != attack['attack']: # Different attacker territory and attacker owner
            first_line += f" através de {attack['attack_itself']}"
        
        first_line += '!\n'

        defend_territories = len(df[df.owner == attack['defend']])
        second_line = f"{attack['defend']} agora possui {defend_territories} territórios.\n" if defend_territories > 0 else f"{attack['defend']} {np.random.choice(ELIMINATIONS)}.\n"

        attack_territories = len(df[df.owner == attack['attack']])
        third_line = f"{attack['attack']} agora possui {attack_territories} territórios.\n"

        fourth_line = f"Ainda restam {len(df.owner.unique())} cidades."

        text = first_line + second_line + third_line + '\n\n' + fourth_line
        f.write(text)
    
    # Comment text (ranking)
    with open("{}/{}/{}.txt".format(basepath, counter, 'comment'), 'w') as f:
        text = 'Cidades já eliminadas:\n\n'
        cities = sorted([(city.nome, city.ranking) for idx, city in df[df.ranking.notnull()].iterrows()], key=lambda x: x[1])
        for name, position in cities:
            text += f"{int(position)}. {name}\n"
        f.write(text)
        
    return



### CHART GENERATION ###

# Read the dataframe
df = read_shapefile()

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
basepath = './figures/'
counter = max([0] + [int(f.split('\\')[-1].split('.')[0].split('_')[-1])
                     for f in glob.glob(basepath + '*')])

# Run the code
while len(df.owner.unique()) > 1:
    # Run the attacks
    attacks = run()
    counter += len(attacks)

    # Create folder
    os.mkdir(basepath + str(counter))
    
    # Plot the map
    fig, ax = plot_map(df, figsize=(15, 12), attack=attacks[-1], fontsize=9)

    # Call Garbage Collector explicitly
    gc.collect()

    # Remove axis, ticks and remove padding
    sns.despine(top=True, right=True, left=True, bottom=True)
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
    ax.set_yticks([])
    ax.set_xticks([])

    # Save post figure
    figure_name = "{}/{}/{}.jpg".format(basepath, counter, 'post')
    plt.savefig(figure_name, dpi=200)
    print("Saved figure to {}".format(figure_name))
    plt.close()

    print("Computing bounds for zoomed image")
    bounds = df[(df.owner == attacks[-1]['attack']) | (df.owner == attacks[-1]['defend'])].geometry.unary_union.bounds
    x_lim = (bounds[0] - 0.02, bounds[2] + 0.02)
    y_lim = (bounds[1] - 0.02, bounds[3] + 0.02)
    fig, ax = plot_map(df,
                       figsize=(15, 12),
                       attack=attacks[-1],
                       fontsize=30,
                       arrow_fontsize=min(x_lim[1] - x_lim[0], y_lim[1] - y_lim[0]) * 200,
                       x_lim=x_lim,
                       y_lim=y_lim,
                       zoom=True)

    # Call Garbage Collector explicitly
    gc.collect()

    # Remove axis, ticks and remove padding
    sns.despine(top=True, right=True, left=True, bottom=True)
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
    ax.set_yticks([])
    ax.set_xticks([])

    # Save figure
    figure_name = "{}/{}/{}.jpg".format(basepath, counter, 'comment')
    plt.savefig(figure_name, dpi=200)
    print("Saved figure to {}".format(figure_name))
    plt.close()

    # Call Garbage Collector explicitly
    gc.collect()    

    # Save post text to file
    save_text(df, basepath, counter, attacks[-1])

    # Update Dataframe JSON file
    print("Saving Updatable Dataframe Information to JSON")
    df[['nome', 'owner', 'protected', 'color', 'ranking']].to_json(
        './guerra.json', orient='records')  # Save to JSON

    # Call Garbage Collector explicitly
    gc.collect()
