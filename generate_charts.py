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
import matplotlib.gridspec as gridspec
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

    print("Generating Geopandas Dataframe")
    df = geopandas.read_file('./Municipio.shp')

    if os.path.isfile('./guerra.json'):  # If already ran, update the fields
        print("Reading special parameters from saved JSON file")
        json_df = pd.read_json('./guerra.json', orient='records')
        df[['owner', 'color', 'protected', 'ranking', 'participated']] = json_df[[
            'owner', 'color', 'protected', 'ranking', 'participated']]
    else:  # Create new fields
        print("Configuring special parameters")
        df['owner'] = df.nome
        df['color'] = [np.random.rand(3,) / 2 + 0.5 for city_name in df.nome]
        df['protected'] = [0 for city_name in df['nome']]
        df['ranking'] = float('nan')
        df['participated'] = False

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


def plot_arrow(a, b, ax, crossed=False, **kwargs):
    '''
    Plot an arrow from city A to city B
    '''

    start = df[df.nome == a].geometry.centroid.values[0]
    end = df[df.nome == b].geometry.centroid.values[0]

    x, y = start.x, start.y
    dx, dy = end.x - x, end.y - y

    # Draw an 'X' in the arrow, if crossed
    if crossed:
        ax.text(x + dx/2, y + dy/2 - 0.002, 'X', va='center', ha='center', zorder=120, fontsize=kwargs.get('arrow_fontsize', 7)) \
           .set_path_effects([PathEffects.withStroke(linewidth=2, foreground='w')])

    return ax.arrow(x, y, dx, dy, width=0.005, head_length=0.02, zorder=119, color='k')


def plot_map(df, x_lim=None, y_lim=None, figsize=(16, 13), attacks=[], city_linewidth=0.03, zoom=False, ax=None, fig=None, **kwargs):
    '''
    Plot map with lim coordinates, and the cities asked with their correspondent color
    '''

    # Configure the plot
    if ax is None or fig is None:
        fig, ax = plt.subplots(figsize=figsize)

    # Plot the cities (using custom code, as the fucking geopandas doesn't work correctly)
    ax.set_aspect('equal')
    collection = PatchCollection([PolygonPatch(poly['geometry'], facecolor=poly['color'] if poly['participated'] else [np.mean(poly['color'])] * 3, hatch='//') for (i, poly) in df.iterrows()], match_original=True)
    ax.add_collection(collection, autolim=True)
    ax.autoscale_view()
    print("Plotted cities")

    # Configure the map size
    if x_lim is not None and y_lim is not None:
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)

    # Plot texts
    owners = df.owner.unique()
    np.random.shuffle(owners)  # Shuffle in place

    # Choose what names to plot, based if we are zoomed in or not
    if zoom: # Only plot zoomed to cities names, assuming there is one attack to zoom at
        plot_names = [owner_name for owner_name in [attacks[0]['attack'], attacks[0]['defend']] if owner_name in owners]
    else:
    # Predefined first owners (always plotted, if alive)
    ALWAYS_ON_TOP = [
                        'Porto Alegre',
                        *[attack['attack'] for attack in attacks],
                        *[attack['defend'] for attack in attacks]
                    ]
    for city in ALWAYS_ON_TOP:
        if city in owners:
            owners = np.concatenate(([city], owners))

        plot_names = owners

    texts_bb = []
    for owner_name in owners:
        owned_by = df[df['owner'] == owner_name]
        center = df[df.owner == owner_name].geometry.unary_union.centroid
        txt = ax.text(center.x, center.y, owner_name,
                       va='center', ha='center', zorder=130 if x_lim is not None else 110, fontsize=kwargs.get('fontsize', 6))
        txt.set_path_effects(
            [PathEffects.withStroke(linewidth=3, foreground='w')])
        bb_transformed = txt.get_window_extent(renderer=fig.canvas.get_renderer()) \
                            .transformed(ax.transData.inverted())

        # Check if it is outside of the map
        outside = False
        if (x_lim is not None) and (y_lim is not None):
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
    print("Plotted text")

    # Plot arrow for attack
    for attack in attacks:
        plot_arrow(attack['attack_itself'], attack['defend_itself'], ax,
                   crossed=not attack['success'], alpha=0.8, zorder=100, **kwargs)

        # Add borders to attack city, and defend city
        collection = PatchCollection([PolygonPatch(df[df.nome == attack['defend_itself']].geometry.values[0], fill=None, zorder=200, edgecolor='blue', linewidth=5 if zoom else 2)], match_original=True)
        ax.add_collection(collection, autolim=True)

        collection = PatchCollection([PolygonPatch(df[df.owner == attack['attack']].geometry.unary_union, fill=None, zorder=200, edgecolor='red', linewidth=2.5 if zoom else 1.5)], match_original=True)
        ax.add_collection(collection, autolim=True)

        # Add borders to defend city owner if it still has territories
        if len(df[df.owner == attack['defend']]) > 0:
            collection = PatchCollection([PolygonPatch(df[df.owner == attack['defend']].geometry.unary_union, fill=None, zorder=200, edgecolor='green', linewidth=1.5 if zoom else 1)], match_original=True)
            ax.add_collection(collection, autolim=True)

        ax.autoscale_view()
    print("Plotted arrows")

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
            attack_city_owner = df[df.nome == attack_city.owner.values[0]]

            # Get the attacking city name
            attack_name = attack_city.nome.values[0]

            # Get the distance to the other cities
            distance_to_cities = DISTANCES[attack_name]

            # Get the names from all cities under the same owner
            owned_by_same = df[df.owner == attack_city_owner.nome.values[0]].nome.values

            # Get the name of the first city, in distance, which is not owned by the same guy
            defend_name = next(
                city[1] for city in distance_to_cities if city[1] not in owned_by_same and city[1])

            # Get its correspondent reference
            defend_city = df[df.nome == defend_name]

            # Get its owner
            defend_city_owner = df[df.nome == defend_city.owner.values[0]]

            # Check city protection
            if defend_city.protected.values[0] > 0:
                # Print information
                print("{} tried to conquer {} from {} through {}, but failed".format(attack_city_owner.nome.values[0], defend_name, defend_city_owner.nome.values[0], attack_name))

                cities.append({'attack': attack_city_owner.nome.values[0],
                               'defend': defend_city_owner.nome.values[0],
                               'attack_itself': attack_name,
                               'defend_itself': defend_name,
                               'success': False})
            else:
                # Print information
                print("{} conquers {} from {} through {}".format(attack_city_owner.nome.values[0], defend_name, defend_city_owner.nome.values[0], attack_name))

                # Update the city owner, color and participation
                df.loc[df['nome'] == defend_name, ['owner']] = attack_city_owner.nome.values[0]
                df.loc[df['nome'] == defend_name, ['color']] = attack_city.color.values
                df.loc[(df['nome'] == defend_name) | (df['nome'] == attack_name), 'participated'] = True

                cities.append({'attack': attack_city_owner.nome.values[0],
                               'defend': defend_city_owner.nome.values[0],
                               'attack_itself': attack_name,
                               'defend_itself': defend_name,
                               'success': True})

                # Verify if need to update ranking
                if defend_city_owner.nome.values[0] not in list(df.owner):
                    df.loc[df.nome == defend_city_owner.nome.values[0], ['ranking']] = len(df.owner.unique()) + 1

            # Decrease protectedness
            df.loc[df['protected'] > 0, ['protected']
                   ] = df[df['protected'] > 0].protected - 1
    return cities


def save_text(df, basepath, counter, attacks):
    '''
        Save the tet for this attack to file
    '''
    LOCATIONS = np.array(['da querência de', 'do território de', ', pertencente às terras de', 'da gurizada de'])
    LOCATIONS_FULL = np.array(['A querência de', 'O exército de', 'A peonada de', 'A gurizada de'])
    CONQUER_VERB = np.array(['ataca', 'derruba', 'passa por cima de'])
    ELIMINATIONS = np.array(['está fora do jogo', 'foi brutalmente eliminado', 'está fora de combate', 'perdeu todos seus territórios', 'foi eliminado', 'tá fora da peleia'])
    MONTHS = ['Janeiro', 'Fevereiro', 'Março', 'Abril', 'Maio', 'Junho', 'Julho', 'Agosto', 'Setembro', 'Outubro', 'Novembro', 'Dezembro']

    # Post text
    with open("{}/{}/{}.txt".format(basepath, counter, 'post'), 'w') as f:
        introduction, first_line, second_line, third_line = '', '', '', ''

        attacks_quantity = len(glob.glob(basepath + '*')) - 1 # Quantity of attacks
        introduction = "{} de {}.".format(MONTHS[attacks_quantity % 12], str(int(attacks_quantity / 12) + 2020))
        
        for attack in attacks:
            first_line += f"{np.random.choice(LOCATIONS_FULL) + ' ' if np.random.randint(100) >= 60 else ''}{attack['attack']} {np.random.choice(CONQUER_VERB) if np.random.randint(100) >= 75 else 'conquista'} {attack['defend_itself']}"

            if attack['defend_itself'] != attack['defend']:  # Different defender territory and defender owner
                first_line += f" {np.random.choice(LOCATIONS) if np.random.randint(100) >= 70 else 'de'} {attack['defend']}"

            if attack['attack_itself'] != attack['attack']:  # Different attacker territory and attacker owner
                first_line += f", através de {attack['attack_itself']}"

            first_line += '\n'

            defend_territories = len(df[df.owner == attack['defend']])
            second_line += f"{attack['defend']} passa a ter {defend_territories} território{'s' if defend_territories > 1 else ''}.\n" if defend_territories > 0 else f"{attack['defend']} {np.random.choice(ELIMINATIONS)}.\n"

            attack_territories = len(df[df.owner == attack['attack']])
            third_line += f"{attack['attack']} conquista seu {attack_territories}º território.\n"

        fourth_line = f"Ainda restam {len(df.owner.unique())} cidades."

        text = introduction + '\n' + first_line + '\n' + second_line + '\n' + third_line + '\n\n' + fourth_line
        f.write(text)

    # Comment text (ranking)
    with open("{}/{}/{}.txt".format(basepath, counter, 'comment'), 'w') as f:
        text = 'Cidades com maiores territórios:\n\n'
        owner_counts = df[(df.participated == True)].owner.value_counts().head(10)
        for idx, (owner, count) in enumerate(owner_counts.iteritems()):
            if count >= 1:
                text += f"{idx + 1}. {owner} -> {count} território{'s' if count > 1 else ''}\n"
        f.write(text)

    return

# CHART GENERATION


# Read the dataframe
df = read_shapefile()

# Precompute the distances
DISTANCES = {}
if os.path.isfile('./distances.json'):
    print("Fetching distances from cached file")
    with open('./distances.json', 'r') as f:
        DISTANCES = json.load(f)
    print("Distances fetched")
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
ATTACK_NUMBER = 3
while len(df.owner.unique()) > 1:
    # Run the attacks
    attacks = run(times=ATTACK_NUMBER)
    counter += len(attacks)
    print("We are at attack {}".format(counter))

    # Create folder
    os.mkdir(basepath + str(counter))

    # Plot the map
    fig, ax = plot_map(df, figsize=(15, 12), attacks=attacks, fontsize=9)
    print("Plotted big map")

    # Call Garbage Collector explicitly
    gc.collect()

    # Remove axis, ticks and remove padding
    sns.despine(top=True, right=True, left=True, bottom=True)
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
    ax.set_yticks([])
    ax.set_xticks([])

    # Add background image
    img = plt.imread('background.png')
    ax.imshow(img, extent=[*(ax.get_xlim()), *(ax.get_ylim())], zorder=-100)

    # Save post figure
    print("Saving figure...")
    figure_name = "{}/{}/{}.jpg".format(basepath, counter, 'post')
    plt.savefig(figure_name, dpi=200)
    print("Saved figure to {}".format(figure_name))
    del img, fig, ax
    plt.close()

    # Call Garbage Collector explicitly
    gc.collect()

    print("Starting to plot attacks zoomed map")
    fig_axs, axs = plt.subplots(ncols=ATTACK_NUMBER)
    for idx, attack in enumerate(attacks):
        bounds = df[(df.owner == attack['attack']) | (df.owner == attack['defend'])].geometry.unary_union.bounds
        x_lim = (bounds[0] - 0.02, bounds[2] + 0.02)
        y_lim = (bounds[1] - 0.02, bounds[3] + 0.02)

        print("Plotting zoomed chart {}/{}".format(idx + 1, len(attacks)))
        fig, ax = plot_map(df,
                           figsize=(15, 12),
                           attacks=[attack],
                           fontsize=15,
                           arrow_fontsize=min(x_lim[1] - x_lim[0], y_lim[1] - y_lim[0]) * 75,
                           x_lim=x_lim,
                           y_lim=y_lim,
                           zoom=True,
                           ax=axs[idx],
                           fig=fig_axs)

        # Call Garbage Collector explicitly
        gc.collect()

        # Remove axis, ticks and remove padding
        sns.despine(top=True, right=True, left=True, bottom=True)
        fig.subplots_adjust(left=0, bottom=0, right=1,
                            top=1, wspace=0.1, hspace=0)
        ax.set_yticks([])
        ax.set_xticks([])

    # Save figure
    print("Saving figure...")
    figure_name = "{}/{}/{}.jpg".format(basepath, counter, 'comment')
    plt.savefig(figure_name, dpi=200)
    print("Saved figure to {}".format(figure_name))
    plt.close()

    # Call Garbage Collector explicitly
    gc.collect()

    # Save post and comment text to file
    save_text(df, basepath, counter, attacks)

    # Update Dataframe JSON file
    print("Saving Updatable Dataframe Information to JSON")
    df[['nome', 'owner', 'color', 'protected', 'ranking', 'participated']].to_json(
        './guerra.json', orient='records')  # Save to JSON
    df[['nome', 'owner', 'color', 'protected', 'ranking', 'participated']].to_json(
        './json/guerra.{}.json'.format(counter), orient='records')  # Save to JSON

    # Call Garbage Collector explicitly
    gc.collect()


# PAREI NO 99 DE POSTAR
# PAREI NO 54 DE COMENTAR
